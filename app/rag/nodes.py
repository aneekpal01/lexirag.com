"""Discrete execution nodes for the LangGraph legal research workflow."""

import re
from typing import Any, Optional
from app.clients.nebius import NebiusTokenFactoryClient
from app.clients.qdrant import QdrantClientWrapper
from app.core.config import Settings, get_settings
from app.core.constants import (
    DEFAULT_MIN_SIMILARITY_SCORE,
    DEFAULT_TOP_K_RETRIEVAL,
    LEGAL_DOMAIN_CORPORATE,
    LEGAL_DOMAIN_CRIMINAL,
    LEGAL_DOMAIN_EMPLOYMENT,
    LEGAL_DOMAIN_GENERAL,
    LEGAL_DOMAIN_STARTUP,
    LEGAL_DOMAIN_TAXATION,
    VALID_LEGAL_DOMAINS,
)
from app.core.logging import get_logger
from app.rag.context import build_grounded_context_block, prune_chunks_by_score_margin
from app.rag.state import LegalGraphState

logger = get_logger(__name__)

# Heuristic cues signaling high-complexity Indian statutory/precedent synthesis
COMPLEX_LEGAL_INDICATORS: tuple[str, ...] = (
    "conflict",
    "precedent",
    "retrospective",
    "overrule",
    "cross-border",
    "concomitant",
    "ultra vires",
    "lifting the corporate veil",
    "nclt",
    "nclat",
    "supreme court",
    "incongruity",
    "reconciliation",
    "amalgamation",
    "debenture",
    "gaar",
    "benami",
    "transfer pricing",
    "section 148",
    "prevention of money laundering",
)


def _detect_domain_from_keywords(query: str) -> Optional[str]:
    """Heuristic keyword detection for Indian legal domain categorization."""
    lowered = query.lower()
    if any(k in lowered for k in ["income tax", "gst", "tds", "assessment year", "it act", "customs", "taxation", "cgst", "sgst", "section 148", "advance tax", "deemed dividend"]):
        return LEGAL_DOMAIN_TAXATION
    if any(k in lowered for k in ["company", "companies act", "director", "board meeting", "shareholder", "mca", "roc", "nclt", "nclat", "amalgamation", "merger", "debenture", "lifting the corporate veil", "csr"]):
        return LEGAL_DOMAIN_CORPORATE
    if any(k in lowered for k in ["employee", "workman", "gratuity", "provident fund", "pf", "wage", "industrial dispute", "factories act", "termination of workman", "maternity benefit"]):
        return LEGAL_DOMAIN_EMPLOYMENT
    if any(k in lowered for k in ["startup", "dpiit", "angel tax", "esop", "term sheet", "convertible note", "seed round"]):
        return LEGAL_DOMAIN_STARTUP
    if any(k in lowered for k in ["bns", "crpc", "ipc", "police", "fir", "bail", "arrest", "pmla", "money laundering", "cheque bounce", "section 138"]):
        return LEGAL_DOMAIN_CRIMINAL
    return None


async def classify_query_node(
    state: LegalGraphState,
    nebius_client: NebiusTokenFactoryClient,
    settings: Settings,
) -> dict[str, Any]:
    """
    Classifies inbound legal questions to route between Nemotron Nano and Nemotron Super,
    infers domain if omitted, and configures adaptive retrieval bounds.
    """
    query_text = state.get("query", "").strip()
    explicit_domain = state.get("domain")
    force_complex = state.get("force_complex", False)

    # 1. Determine domain: preserve explicit or infer from heuristics
    domain_inferred = explicit_domain is None or not explicit_domain.strip()
    detected_domain: Optional[str] = explicit_domain if not domain_inferred else _detect_domain_from_keywords(query_text)

    # Helper to calculate adaptive retrieval bounds
    def _get_adaptive_bounds(is_complex_tier: bool) -> tuple[int, float]:
        if is_complex_tier:
            return settings.top_k_complex, settings.min_score_complex
        return settings.top_k_simple, settings.min_score_simple

    # 2. Manual force-override if client explicitly demands high-capacity reasoning
    if force_complex:
        logger.info("Forced complex reasoning requested by client caller")
        top_k, min_score = _get_adaptive_bounds(True)
        return {
            "query_complexity": "complex",
            "selected_model": settings.nemotron_super_model,
            "classification_reasoning": "Client explicitly set force_complex_reasoning flag.",
            "detected_domain": detected_domain or LEGAL_DOMAIN_GENERAL,
            "domain_inferred": domain_inferred,
            "retrieval_top_k": top_k,
            "retrieval_min_score": min_score,
        }

    # 3. Fast heuristic check for overt multi-statute indicators
    lowered_query = query_text.lower()
    matches = [indicator for indicator in COMPLEX_LEGAL_INDICATORS if indicator in lowered_query]
    if len(matches) >= 2:
        logger.info("Heuristic complexity trigger matched indicators: %s", matches)
        top_k, min_score = _get_adaptive_bounds(True)
        return {
            "query_complexity": "complex",
            "selected_model": settings.nemotron_super_model,
            "classification_reasoning": f"Query contains multiple complex legal terms: {', '.join(matches)}",
            "detected_domain": detected_domain or LEGAL_DOMAIN_GENERAL,
            "domain_inferred": domain_inferred,
            "retrieval_top_k": top_k,
            "retrieval_min_score": min_score,
        }

    # 4. LLM Classification & Domain Inference via Nemotron Nano
    classification_prompt = [
        {
            "role": "system",
            "content": (
                "You are an expert Indian Senior Advocate and judicial research classifier. "
                "Analyze the user's legal question under Indian jurisprudence (Acts of Parliament, Rules, Case Law). "
                "Classify the query strictly into one of two tiers:\n"
                "1. 'SIMPLE': Single provision lookup, straightforward statutory thresholds, basic definition, filing fees, or direct statutory penalties.\n"
                "2. 'COMPLEX': Multi-statute interplay, reconciliation of contradictory precedents, structural tax/corporate reorganization, constitutional challenge, or ambiguous statutory interpretations.\n\n"
                "Also identify the primary Indian legal domain among: taxation, corporate_law, employment_labor, startup_compliance, criminal_procedure, general_statutory.\n\n"
                "Output Format:\n"
                "CLASSIFICATION: [SIMPLE or COMPLEX]\n"
                "DOMAIN: [taxation | corporate_law | employment_labor | startup_compliance | criminal_procedure | general_statutory]\n"
                "RATIONALE: [One sentence explanation]"
            ),
        },
        {"role": "user", "content": f"Indian Legal Query: {query_text}"},
    ]

    try:
        classification_result = await nebius_client.create_chat_completion(
            model=settings.nemotron_nano_model,
            messages=classification_prompt,
            temperature=0.1,
            max_tokens=250,
        )

        is_complex = "CLASSIFICATION: COMPLEX" in classification_result.upper() or "COMPLEX" in classification_result.upper().split("\n")[0]
        assigned_complexity = "complex" if is_complex else "simple"
        assigned_model = (
            settings.nemotron_super_model if is_complex else settings.nemotron_nano_model
        )

        # Extract domain from response if not yet determined
        if domain_inferred and not detected_domain:
            for domain_candidate in VALID_LEGAL_DOMAINS:
                if f"DOMAIN: {domain_candidate}".lower() in classification_result.lower() or domain_candidate in classification_result.lower():
                    detected_domain = domain_candidate
                    break

        resolved_domain = detected_domain or LEGAL_DOMAIN_GENERAL
        top_k, min_score = _get_adaptive_bounds(is_complex)

        logger.info(
            "Query classified | assigned: %s | domain: %s (inferred: %s) | top_k: %d | min_score: %.2f | model: %s",
            assigned_complexity,
            resolved_domain,
            domain_inferred,
            top_k,
            min_score,
            assigned_model,
        )

        return {
            "query_complexity": assigned_complexity,
            "selected_model": assigned_model,
            "classification_reasoning": classification_result,
            "detected_domain": resolved_domain,
            "domain_inferred": domain_inferred,
            "retrieval_top_k": top_k,
            "retrieval_min_score": min_score,
        }

    except Exception as exc:
        logger.warning(
            "Classification node encountered error; defaulting conservatively to Super-120b: %s",
            exc,
        )
        top_k, min_score = _get_adaptive_bounds(True)
        return {
            "query_complexity": "complex",
            "selected_model": settings.nemotron_super_model,
            "classification_reasoning": f"Fallback to complex tier due to classifier error: {exc}",
            "detected_domain": detected_domain or LEGAL_DOMAIN_GENERAL,
            "domain_inferred": domain_inferred,
            "retrieval_top_k": top_k,
            "retrieval_min_score": min_score,
        }


async def retrieve_context_node(
    state: LegalGraphState,
    nebius_client: NebiusTokenFactoryClient,
    qdrant_wrapper: QdrantClientWrapper,
    settings: Settings,
) -> dict[str, Any]:
    """
    Retrieves grounding legal provisions and case law precedents from Qdrant Cloud.
    
    Applies adaptive top-k and min-score thresholds, compound filtering on domain and
    document_id, strict filter relaxation rules, and relative score margin pruning.
    """
    query_text = state["query"]
    domain_to_filter = state.get("detected_domain") or state.get("domain")
    document_id_filter = state.get("document_id")
    domain_inferred = state.get("domain_inferred", False)

    top_k = state.get("retrieval_top_k", settings.top_k_retrieval_limit)
    min_score = state.get("retrieval_min_score", settings.min_similarity_score)

    # Step 1: Generate query embedding vector via Nebius Token Factory
    query_vector = await nebius_client.create_embedding(text=query_text)

    # Step 2: Vector similarity search against pre-embedded corpus in Qdrant Cloud
    retrieved_statutes = await qdrant_wrapper.search_statutes(
        query_vector=query_vector,
        top_k=top_k,
        min_score=min_score,
        domain_filter=domain_to_filter,
        document_id_filter=document_id_filter,
    )

    filter_relaxed = False

    # Step 3: Enforce strict Filter Relaxation Safety Rules (Mandatory Correction #1)
    if not retrieved_statutes:
        # If user explicitly provided document_id: NEVER relax!
        if document_id_filter and document_id_filter.strip():
            logger.info(
                "Zero chunks retrieved for explicit document_id '%s'. Preserving strict document boundary; not relaxing.",
                document_id_filter,
            )
        # If domain was automatically inferred (and no doc_id was passed), execute controlled relaxation
        elif domain_inferred and domain_to_filter:
            logger.info(
                "Zero chunks retrieved with auto-inferred domain filter '%s'. Executing controlled filter relaxation.",
                domain_to_filter,
            )
            relaxed_statutes = await qdrant_wrapper.search_statutes(
                query_vector=query_vector,
                top_k=top_k,
                min_score=min_score,
                domain_filter=None,
                document_id_filter=None,
            )
            if relaxed_statutes:
                retrieved_statutes = relaxed_statutes
                filter_relaxed = True
                logger.info("Filter relaxation succeeded: recovered %d statutory chunks", len(retrieved_statutes))
            else:
                filter_relaxed = True
        else:
            logger.info("Zero chunks retrieved for explicit domain filter '%s'; preserving client filter.", domain_to_filter)

    serialized_chunks = [chunk.model_dump() for chunk in retrieved_statutes]

    # Step 4: Relative Score Margin Pruning
    filtered_chunks = prune_chunks_by_score_margin(
        serialized_chunks,
        margin_ratio=settings.score_margin_ratio,
        min_score=min_score,
    )

    fallback_triggered = len(filtered_chunks) == 0

    if fallback_triggered:
        logger.warning(
            "Zero statutory records survived filtering and score margin pruning (floor %.2f) for query: %s",
            min_score,
            query_text,
        )

    return {
        "query_embedding": query_vector,
        "retrieved_chunks": serialized_chunks,
        "filtered_chunks": filtered_chunks,
        "fallback_triggered": fallback_triggered,
        "filter_relaxed": filter_relaxed,
    }


async def generate_answer_node(
    state: LegalGraphState,
    nebius_client: NebiusTokenFactoryClient,
    settings: Optional[Settings] = None,
) -> dict[str, Any]:
    """
    Synthesizes the legal opinion using the assigned Nemotron reasoning tier.
    
    Applies strict evidence-first grounding rules and anti-hallucination guardrails.
    Never invents sections or answers from memory when retrieval evidence is absent.
    """
    query_text = state["query"]
    selected_model = state["selected_model"]
    current_settings = settings or get_settings()

    # Prefer filtered_chunks (pruned/deduplicated), fallback to retrieved_chunks for backward compatibility
    candidate_chunks = state.get("filtered_chunks")
    if candidate_chunks is None:
        candidate_chunks = state.get("retrieved_chunks", [])

    fallback_triggered = state.get("fallback_triggered", False)

    # Compile statutory exhibition context block with character budgeting
    compiled_context, included_chunks = build_grounded_context_block(
        candidate_chunks,
        max_chars=current_settings.max_context_chars,
    )

    system_instruction = (
        "You are LexiRAG, an authoritative Indian senior legal research counsel specializing in Indian "
        "Corporate, Tax, Labor, and Commercial Law. Your audience consists of Advocates, Chartered Accountants, "
        "and In-House Legal Counsel.\n\n"
        "STRICT GROUNDING DIRECTIVES (EVIDENCE-FIRST PRINCIPLE):\n"
        "1. EXCLUSIVE RELIANCE ON EXHIBITS: Ground your legal opinion exclusively on the provisions and facts provided "
        "in the numbered exhibits below ([EXHIBIT 1], [EXHIBIT 2], etc.).\n"
        "2. PROHIBITION OF SPECULATION: Never invent, assume, or speculate regarding statutory Section numbers, sub-sections, "
        "or penalty figures not explicitly evidenced in the exhibits.\n"
        "3. SAFE REFUSAL ON ABSENT EVIDENCE: If no exhibits are provided or if the indexed corpus contains insufficient evidence, "
        "explicitly declare that the indexed corpus does not contain sufficient statutory evidence to answer the question, "
        "and decline to fabricate statutory provisions.\n"
        "4. EXPLICIT EVIDENCE BOUNDARIES: If the provided exhibits partially answer the query, clearly demarcate what is "
        "substantiated by the evidence from what remains unsupported.\n"
        "5. FORMAL STRUCTURE: (a) Executive Conclusion, (b) Statutory / Contractual Analysis grounded in Exhibits, "
        "(c) Compliance Implications & Practical Risks, (d) Evidentiary Scope & Limitations."
    )

    if fallback_triggered or not included_chunks:
        user_message_content = (
            f"STATUTORY & DOCUMENT EVIDENCE CONTEXT:\n{compiled_context}\n\n"
            f"CLIENT LEGAL QUERY:\n{query_text}\n\n"
            "INSTRUCTION: Since the retrieved corpus yielded zero matching statutory provisions for this query, "
            "provide an explicit statement declaring the lack of sufficient indexed evidence. "
            "Do NOT fabricate section numbers or provisions from memory."
        )
    else:
        user_message_content = (
            f"STATUTORY & DOCUMENT EVIDENCE CONTEXT:\n{compiled_context}\n\n"
            f"CLIENT LEGAL QUERY:\n{query_text}\n\n"
            "Provide your grounded legal opinion based strictly on the exhibits above:"
        )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_message_content},
    ]

    synthesized_answer = await nebius_client.create_chat_completion(
        model=selected_model,
        messages=messages,
        temperature=0.1 if (fallback_triggered or not included_chunks) else 0.2,
        max_tokens=3500,
    )

    return {
        "raw_answer": synthesized_answer,
        "context_text": compiled_context,
        "filtered_chunks": included_chunks,
    }


async def format_citations_node(
    state: LegalGraphState,
) -> dict[str, Any]:
    """
    Validates, deduplicates, and formats citations into the final response payload.
    
    Aligns citations strictly with the evidence chunks actually included in the generation context.
    """
    raw_answer = state.get("raw_answer", "")
    candidate_chunks = state.get("filtered_chunks")
    if candidate_chunks is None:
        candidate_chunks = state.get("retrieved_chunks", [])

    formatted_citations: list[dict[str, Any]] = []
    seen_provisions: set[str] = set()

    for chunk in candidate_chunks:
        act_or_doc = (
            chunk.get("act_name")
            or chunk.get("document_name")
            or chunk.get("document_id")
            or "Unknown Statute"
        ).strip()
        section = (chunk.get("section") or chunk.get("heading") or "General Provision").strip()
        unique_key = f"{act_or_doc.lower()}::{section.lower()}"

        if unique_key in seen_provisions:
            continue
        seen_provisions.add(unique_key)

        formatted_citations.append(
            {
                "act_name": act_or_doc,
                "section": section,
                "sub_section": chunk.get("sub_section"),
                "title": chunk.get("title") or chunk.get("heading"),
                "court_or_authority": chunk.get("court_or_authority"),
                "citation_ref": chunk.get("citation_ref"),
                "relevance_excerpt": chunk.get("content", "")[:300] + ("..." if len(chunk.get("content", "")) > 300 else ""),
                "similarity_score": round(float(chunk.get("similarity_score", 0.0)), 4),
                "document_id": chunk.get("document_id"),
                "document_name": chunk.get("document_name"),
                "page_number": chunk.get("page_number"),
                "heading": chunk.get("heading"),
                "chunk_id": chunk.get("chunk_id"),
            }
        )

    return {
        "citations": formatted_citations,
        "final_answer": raw_answer,
    }

