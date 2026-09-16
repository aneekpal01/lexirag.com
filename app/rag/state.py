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
    jurisdiction: str
    force_complex: bool

    # Classification results
    query_complexity: QueryComplexity
    selected_model: str
    classification_reasoning: Optional[str]

    # Vector retrieval artifacts
    query_embedding: list[float]
    retrieved_chunks: list[dict[str, Any]]
    fallback_triggered: bool

    # Generation & reasoning outputs
    raw_answer: str
    juridical_reasoning: Optional[str]

    # Citation formatting outputs
    citations: list[dict[str, Any]]
    final_answer: str

    # Observability & diagnostic tracing
    execution_latency_ms: float
    error_context: Optional[str]
