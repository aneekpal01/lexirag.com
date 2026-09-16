"""LangGraph state representation for the Indian Legal Research pipeline."""

from typing import Any, Optional, TypedDict
from app.schemas.query import QueryComplexity


class LegalGraphState(TypedDict, total=False):
    """
    Mutable state object passed between nodes in the LangGraph workflow.
    
    Adheres strictly to LangGraph TypedDict state semantics.
    """
    # Inbound query specifications
    query: str
    domain: Optional[str]
    document_id: Optional[str]
    jurisdiction: str
    force_complex: bool

    # Classification & routing results
    query_complexity: QueryComplexity
    selected_model: str
    classification_reasoning: Optional[str]
    detected_domain: Optional[str]
    domain_inferred: bool
    retrieval_top_k: int
    retrieval_min_score: float

    # Vector retrieval & evidence artifacts
    query_embedding: list[float]
    retrieved_chunks: list[dict[str, Any]]
    filtered_chunks: list[dict[str, Any]]
    context_text: str
    fallback_triggered: bool
    filter_relaxed: bool

    # Phase 3 Evidence Graph & Expansion Artifacts
    expanded_chunks: list[dict[str, Any]]
    evidence_graph: Optional[dict[str, Any]]
    expansion_applied: bool
    expansion_count: int

    # Generation & reasoning outputs
    raw_answer: str
    juridical_reasoning: Optional[str]

    # Citation formatting outputs
    citations: list[dict[str, Any]]
    final_answer: str

    # Observability & diagnostic tracing
    execution_latency_ms: float
    error_context: Optional[str]
