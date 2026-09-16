"""Discrete execution nodes for the LangGraph legal research workflow."""

import re
from typing import Any
from app.clients.nebius import NebiusTokenFactoryClient
from app.clients.qdrant import QdrantClientWrapper
from app.core.config import Settings
from app.core.constants import (
    DEFAULT_MIN_SIMILARITY_SCORE,
    DEFAULT_TOP_K_RETRIEVAL,
)
from app.core.logging import get_logger
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


async def classify_query_node(
    state: LegalGraphState,
    nebius_client: NebiusTokenFactoryClient,
    settings: Settings,
) -> dict[str, Any]:
    """
    Classifies inbound legal questions to route between Nemotron Nano and Nemotron Super.
    
    Categorization Rules:
    - 'simple': Definitional inquiries, single-section statutory thresholds, standard penalty
      schedules, filing timelines, or direct compliance lookups.
      -> Handled efficiently by nvidia/nemotron-3-nano-30b-a3b.
    - 'complex': Interplay between conflicting statutes (e.g., IBC vs Companies Act vs SARFAESI),
      interpretative ambiguities, jurisdictional conflicts, tax litigation, or precedent reconciliation.
      -> Routed to nvidia/nemotron-3-super-120b-a12b for multi-hop legal reasoning.
    """
    query_text = state.get("query", "").strip()

    # Manual force-override if client explicitly demands high-capacity reasoning
    if state.get("force_complex", False):
        logger.info("Forced complex reasoning requested by client caller")
        return {
            "query_complexity": "complex",
            "selected_model": settings.nemotron_super_model,
            "classification_reasoning": "Client explicitly set force_complex_reasoning flag.",
        }

    # Fast heuristic check for overt multi-statute indicators
    lowered_query = query_text.lower()
    matches = [indicator for indicator in COMPLEX_LEGAL_INDICATORS if indicator in lowered_query]
    if len(matches) >= 2:
        logger.info("Heuristic complexity trigger matched indicators: %s", matches)
        return {
            "query_complexity": "complex",
            "selected_model": settings.nemotron_super_model,
            "classification_reasoning": f"Query contains multiple complex legal terms: {', '.join(matches)}",
        }

    classification_prompt = [
        {
            "role": "system",
            "content": (
                "You are an expert Indian Senior Advocate and judicial research classifier. "
                "Analyze the user's legal question under Indian jurisprudence (Acts of Parliament, Rules, Case Law). "
                "Classify the query strictly into one of two tiers:\n"
                "1. 'SIMPLE': Single provision lookup, straightforward statutory thresholds, basic definition, filing fees, or direct statutory penalties.\n"
                "2. 'COMPLEX': Multi-statute interplay, reconciliation of contradictory precedents, structural tax/corporate reorganization, constitutional challenge, or ambiguous statutory interpretations.\n\n"
                "Output Format:\n"
                "CLASSIFICATION: [SIMPLE or COMPLEX]\n"
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

        logger.info(
            "Query classified | assigned: %s | model: %s",
            assigned_complexity,
            assigned_model,
        )

        return {
            "query_complexity": assigned_complexity,
            "selected_model": assigned_model,
            "classification_reasoning": classification_result,
        }

    except Exception as exc:
        logger.warning(
            "Classification node encountered error; defaulting conservatively to Super-120b: %s",
            exc,
        )
        return {
            "query_complexity": "complex",
            "selected_model": settings.nemotron_super_model,
            "classification_reasoning": f"Fallback to complex tier due to classifier error: {exc}",
        }


async def retrieve_context_node(
    state: LegalGraphState,
    nebius_client: NebiusTokenFactoryClient,
    qdrant_wrapper: QdrantClientWrapper,
    settings: Settings,
) -> dict[str, Any]:
    """
    Retrieves grounding legal provisions and case law precedents from Qdrant Cloud.
    
    Generates dense embeddings with BAAI/bge-m3 via Nebius Token Factory and handles
    empty retrieval outcomes without terminating the pipeline.
    """
    query_text = state["query"]
    domain_hint = state.get("domain")

    # Step 1: Generate query embedding vector via Nebius Token Factory
    query_vector = await nebius_client.create_embedding(text=query_text)

    # Step 2: Vector similarity search against pre-embedded corpus in Qdrant Cloud
    retrieved_statutes = await qdrant_wrapper.search_statutes(
        query_vector=query_vector,
        top_k=settings.top_k_retrieval_limit,
        min_score=settings.min_similarity_score,
        domain_filter=domain_hint,
    )

    if not retrieved_statutes:
        logger.warning(
            "Zero statutory records retrieved surpassing similarity threshold %.2f for query: %s",
            settings.min_similarity_score,
            query_text,
        )
        return {
            "query_embedding": query_vector,
            "retrieved_chunks": [],
            "fallback_triggered": True,
        }

    serialized_chunks = [chunk.model_dump() for chunk in retrieved_statutes]
    return {
        "query_embedding": query_vector,
        "retrieved_chunks": serialized_chunks,
        "fallback_triggered": False,
    }


async def generate_answer_node(
    state: LegalGraphState,
    nebius_client: NebiusTokenFactoryClient,
) -> dict[str, Any]:
    """
    Synthesizes the legal opinion using the assigned Nemotron reasoning tier.
    
    Constructs an Indian legal context block with exact statutory sections and provisions.
    Extracts output from `reasoning_content` (with fallback to `content`).
    """
    query_text = state["query"]
    selected_model = state["selected_model"]
    retrieved_chunks = state.get("retrieved_chunks", [])
    fallback_triggered = state.get("fallback_triggered", False)

    # Compile statutory context block
    if retrieved_chunks:
        context_segments = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            act_header = f"[{index}] {chunk['act_name']}, {chunk['section']}"
            if chunk.get("title"):
                act_header += f" - {chunk['title']}"
            if chunk.get("court_or_authority"):
                act_header += f" ({chunk['court_or_authority']}"
                if chunk.get("citation_ref"):
                    act_header += f", {chunk['citation_ref']}"
                act_header += ")"

            context_segments.append(
                f"{act_header}\nVERBATIM PROVISION:\n{chunk['content']}\n"
            )
        compiled_context = "\n---\n".join(context_segments)
    else:
        compiled_context = (
            "No direct statutory chunks met the similarity threshold in the legal vector index. "
            "Proceed based on core Indian codified statutes, general principles of jurisprudence, "
            "and explicitly state that exact section verification from the Official Gazette is recommended."
        )

    system_instruction = (
        "You are LexiRAG, a senior legal research counsel specializing in Indian Corporate, Tax, "
        "Regulatory, and Commercial Law. Your audience comprises Indian Advocates, Chartered Accountants, "
        "and General Counsels.\n\n"
        "Guidelines for Indian Legal Synthesis:\n"
        "1. GROUNDING: Anchor every assertion in Indian statutes (e.g., Companies Act 2013, Income Tax Act 1961, "
        "GST Acts 2017, IBC 2016, BNS 2023, FEMA 1999) using the provided context.\n"
        "2. SPECIFICITY: Cite specific Section numbers, Sub-sections, Provsios, and Explanation clauses.\n"
        "3. STRUCTURE: Provide: (a) Executive Summary / Direct Conclusion, (b) Statutory Analysis & Relevant Provisions, "
        "(c) Compliance Implications / Practical Risks, (d) Formal Citations.\n"
        "4. TONE: Authoritative, formal, and legally rigorous. Do not use conversational filler."
    )

    user_message_content = (
        f"STATUTORY & PRECEDENT CONTEXT:\n{compiled_context}\n\n"
        f"CLIENT LEGAL QUERY:\n{query_text}\n\n"
        "Provide your grounded legal opinion:"
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_message_content},
    ]

    # Dispatch to the chosen Nemotron model on Nebius Token Factory
    synthesized_answer = await nebius_client.create_chat_completion(
        model=selected_model,
        messages=messages,
        temperature=0.2,
        max_tokens=3500,
    )

    return {
        "raw_answer": synthesized_answer,
    }


async def format_citations_node(
    state: LegalGraphState,
) -> dict[str, Any]:
    """
    Validates, deduplicates, and formats citations into the final response payload.
    
    Aligns retrieved chunks with verified statutory provisions.
    """
    raw_answer = state.get("raw_answer", "")
    retrieved_chunks = state.get("retrieved_chunks", [])

    formatted_citations: list[dict[str, Any]] = []
    seen_provisions: set[str] = set()

    for chunk in retrieved_chunks:
        unique_key = f"{chunk['act_name']}::{chunk['section']}"
        if unique_key in seen_provisions:
            continue
        seen_provisions.add(unique_key)

        formatted_citations.append(
            {
                "act_name": chunk["act_name"],
                "section": chunk["section"],
                "sub_section": chunk.get("sub_section"),
                "title": chunk.get("title"),
                "court_or_authority": chunk.get("court_or_authority"),
                "citation_ref": chunk.get("citation_ref"),
                "relevance_excerpt": chunk["content"][:300] + ("..." if len(chunk["content"]) > 300 else ""),
                "similarity_score": round(chunk["similarity_score"], 4),
            }
        )

    return {
        "citations": formatted_citations,
        "final_answer": raw_answer,
    }
