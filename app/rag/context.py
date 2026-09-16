"""Context construction, evidence deduplication, and formatting for LexiRAG."""

from collections import OrderedDict
from typing import Any, Optional
from app.core.constants import DEFAULT_MAX_CONTEXT_CHARS, DEFAULT_SCORE_MARGIN_RATIO
from app.core.logging import get_logger

logger = get_logger(__name__)


def prune_chunks_by_score_margin(
    chunks: list[dict[str, Any]],
    margin_ratio: float = DEFAULT_SCORE_MARGIN_RATIO,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Prunes retrieved chunks based on relative score dropoff from the top candidate.
    
    Initial heuristic: A candidate is retained if:
        chunk.similarity_score >= max(min_score, top_chunk.similarity_score * margin_ratio)
    
    Guarantees retention of the top result as long as it meets min_score.
    """
    if not chunks:
        return []

    # Sort descending by similarity score to guarantee index 0 is top hit
    sorted_chunks = sorted(chunks, key=lambda c: float(c.get("similarity_score", 0.0)), reverse=True)
    top_score = float(sorted_chunks[0].get("similarity_score", 0.0))

    if top_score < min_score:
        logger.info("Top chunk score %.4f is below minimum threshold %.4f; pruning all", top_score, min_score)
        return []

    cutoff_threshold = max(min_score, top_score * margin_ratio)
    pruned: list[dict[str, Any]] = [
        c for c in sorted_chunks if float(c.get("similarity_score", 0.0)) >= cutoff_threshold
    ]

    logger.debug(
        "Score margin pruning | input: %d | retained: %d | top_score: %.4f | cutoff: %.4f (ratio: %.2f)",
        len(chunks),
        len(pruned),
        top_score,
        cutoff_threshold,
        margin_ratio,
    )
    return pruned


def deduplicate_evidence_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Deduplicates statutory and document evidence chunks before generation context assembly.
    
    Keyed by normalized (document_or_act, section_or_heading).
    When duplicates exist:
    - Retains primary evidence role over supporting role.
    - Retains the highest similarity score if present.
    - Merges unique supplementary text if distinct.
    - Preserves metadata (document_id, page_number, heading, domain, expansion_reason).
    """
    if not chunks:
        return []

    deduped_map: OrderedDict[str, dict[str, Any]] = OrderedDict()

    for chunk in chunks:
        act_or_doc = (
            chunk.get("act_name")
            or chunk.get("document_name")
            or chunk.get("document_id")
            or "Unknown Source"
        ).strip()
        section = (
            chunk.get("section")
            or chunk.get("heading")
            or "General Provision"
        ).strip()

        key = f"{act_or_doc.lower()}::{section.lower()}"

        if key not in deduped_map:
            # First instance: store a shallow copy
            deduped_map[key] = dict(chunk)
        else:
            existing = deduped_map[key]

            # Primary role always supersedes supporting role
            if chunk.get("evidence_role") == "primary":
                existing["evidence_role"] = "primary"

            # Retain valid similarity score
            new_score = chunk.get("similarity_score")
            existing_score = existing.get("similarity_score")
            if new_score is not None:
                if existing_score is None or float(new_score) > float(existing_score):
                    existing["similarity_score"] = float(new_score)

            # If the duplicate chunk contains non-overlapping text, append it
            existing_content = existing.get("content", "").strip()
            new_content = chunk.get("content", "").strip()
            if new_content and new_content not in existing_content:
                existing["content"] = f"{existing_content}\n\n[Additional Excerpt]:\n{new_content}"

            # Preserve page numbers, sub-sections, or expansion details if missing in original
            if not existing.get("page_number") and chunk.get("page_number"):
                existing["page_number"] = chunk.get("page_number")
            if not existing.get("sub_section") and chunk.get("sub_section"):
                existing["sub_section"] = chunk.get("sub_section")
            if not existing.get("expansion_reason") and chunk.get("expansion_reason"):
                existing["expansion_reason"] = chunk.get("expansion_reason")

    deduped_list = list(deduped_map.values())
    logger.debug("Evidence deduplication | input: %d | unique: %d", len(chunks), len(deduped_list))
    return deduped_list


def group_evidence_hierarchically(chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Groups evidence chunks by their parent Statute or Document name."""
    grouped: dict[str, list[dict[str, Any]]] = OrderedDict()
    for chunk in chunks:
        source_key = (
            chunk.get("act_name")
            or chunk.get("document_name")
            or "Statutory Authority"
        ).strip()
        if source_key not in grouped:
            grouped[source_key] = []
        grouped[source_key].append(chunk)
    return grouped


def build_grounded_context_block(
    chunks: list[dict[str, Any]],
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Assembles evidence chunks into structured, exhibition-labeled context for Nemotron.
    
    Priority Budgeting: Primary evidence receives guaranteed priority; supporting
    neighbor context consumes remaining character headroom only.
    Returns:
        (compiled_context_string, list_of_retained_chunks_used_in_context)
    """
    if not chunks:
        return (
            "NO VERIFIED STATUTORY OR DOCUMENT EVIDENCE FOUND IN THE INDEXED CORPUS.\n"
            "The system retrieved zero matching provisions above the required similarity threshold.",
            [],
        )

    deduped = deduplicate_evidence_chunks(chunks)

    # Order candidates: primary evidence first, followed by supporting context
    primary_chunks = [c for c in deduped if c.get("evidence_role") == "primary" or not c.get("evidence_role")]
    supporting_chunks = [c for c in deduped if c.get("evidence_role") and c.get("evidence_role") != "primary"]
    ordered_candidates = primary_chunks + supporting_chunks

    exhibit_index = 1
    exhibit_blocks: list[str] = []
    included_chunks: list[dict[str, Any]] = []
    current_char_count = 0

    for chunk in ordered_candidates:
        source_name = (
            chunk.get("act_name")
            or chunk.get("document_name")
            or "Statutory Authority"
        ).strip()
        section_label = chunk.get("section") or chunk.get("heading") or "General Provision"
        role_val = chunk.get("evidence_role", "primary")

        # Determine human-readable role and score representation
        if role_val == "supporting_prev":
            role_desc = "Supporting Context (Sequential Predecessor)"
            score_desc = "Structural Context (No Vector Score)"
        elif role_val == "supporting_next":
            role_desc = "Supporting Context (Sequential Successor)"
            score_desc = "Structural Context (No Vector Score)"
        elif role_val != "primary":
            role_desc = "Supporting Context"
            score_desc = "Structural Context (No Vector Score)"
        else:
            role_desc = "Primary Evidence"
            raw_score = chunk.get("similarity_score")
            score_desc = f"{float(raw_score):.4f}" if raw_score is not None else "Direct Match"

        header_lines = [f"[EXHIBIT {exhibit_index}] Source: {source_name} | Provision: {section_label}"]

        meta_details = [f"Role: {role_desc}"]
        if chunk.get("title"):
            meta_details.append(f"Title: {chunk['title']}")
        if chunk.get("page_number"):
            meta_details.append(f"Page: {chunk['page_number']}")
        if chunk.get("domain"):
            meta_details.append(f"Domain: {chunk['domain']}")
        meta_details.append(f"Match Score: {score_desc}")
        if chunk.get("expansion_reason"):
            meta_details.append(f"Expansion Note: {chunk['expansion_reason']}")

        if meta_details:
            header_lines.append(" | ".join(meta_details))

        header = "\n".join(header_lines)
        content = chunk.get("content", "").strip()
        block = f"{header}\nVERBATIM PROVISION:\n{content}\n"

        block_len = len(block)
        if current_char_count + block_len > max_chars and included_chunks:
            logger.warning(
                "Context character budget reached (%d / %d chars). Truncating remaining %d exhibits (role: %s).",
                current_char_count,
                max_chars,
                len(ordered_candidates) - len(included_chunks),
                role_desc,
            )
            break

        exhibit_blocks.append(block)
        included_chunks.append(chunk)
        current_char_count += block_len
        exhibit_index += 1

    compiled_context = "\n---\n".join(exhibit_blocks)
    return compiled_context, included_chunks
