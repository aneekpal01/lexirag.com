"""Request and response schemas for the legal query pipeline."""

from typing import Literal, Optional
from pydantic import BaseModel, Field

from app.core.constants import (
    MAX_QUERY_CHARACTER_LENGTH,
    MIN_QUERY_CHARACTER_LENGTH,
    VALID_LEGAL_DOMAINS,
)

QueryComplexity = Literal["simple", "complex"]


class LegalCitation(BaseModel):
    """Structured statutory or judicial precedent citation extracted from context."""

    act_name: str = Field(
        ...,
        description="Formal title of the statute or central regulation (e.g., 'Companies Act, 2013').",
        examples=["Companies Act, 2013"],
    )
    section: str = Field(
        ...,
        description="Specific statutory section, rule, or regulation number.",
        examples=["Section 185"],
    )
    sub_section: Optional[str] = Field(
        default=None,
        description="Sub-section or clause designation if applicable.",
        examples=["(1)(a)"],
    )
    title: Optional[str] = Field(
        default=None,
        description="Statutory heading or case title.",
        examples=["Loans to directors, etc."],
    )
    court_or_authority: Optional[str] = Field(
        default=None,
        description="Adjudicating body (e.g., 'Supreme Court of India', 'NCLAT', 'ITAT').",
        examples=["Supreme Court of India"],
    )
    citation_ref: Optional[str] = Field(
        default=None,
        description="Formal law report citation if a judicial ruling.",
        examples=["(2021) 4 SCC 123"],
    )
    relevance_excerpt: str = Field(
        ...,
        description="Direct verbatim snippet from the retrieved statutory text grounding the advice.",
    )
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score produced by BGE-M3 dense vector search.",
    )


class LegalQueryRequest(BaseModel):
    """Inbound client payload requesting Indian legal research analysis."""

    query: str = Field(
        ...,
        min_length=MIN_QUERY_CHARACTER_LENGTH,
        max_length=MAX_QUERY_CHARACTER_LENGTH,
        description="Legal query regarding Indian corporate, tax, labor, or regulatory statutory provisions.",
        examples=[
            "What are the statutory conditions under Section 185 of the Companies Act 2013 for advancing loans to a private company having common directors?"
        ],
    )
    domain: Optional[str] = Field(
        default=None,
        description="Optional domain filter hint (e.g., 'corporate_law', 'taxation', 'employment_labor').",
    )
    jurisdiction: str = Field(
        default="India",
        description="Applicable jurisdiction (defaults to Republic of India).",
    )
    force_complex_reasoning: bool = Field(
        default=False,
        description="Force execution through Nemotron-3-Super-120b regardless of classification heuristics.",
    )


class LegalQueryResponse(BaseModel):
    """Outbound API response returning grounded legal opinion and formal citations."""

    answer: str = Field(
        ...,
        description="Comprehensive, statutory grounded legal answer authored by Nemotron reasoning model.",
    )
    citations: list[LegalCitation] = Field(
        default_factory=list,
        description="List of verified statutory provisions or precedents supporting the opinion.",
    )
    complexity: QueryComplexity = Field(
        ...,
        description="Assigned complexity classification determining model tier.",
    )
    model_used: str = Field(
        ...,
        description="Nebius Token Factory model identifier utilized for final generation.",
    )
    retrieved_chunks_count: int = Field(
        ...,
        ge=0,
        description="Total statutory document segments surfaced from Qdrant vector store.",
    )
    execution_time_ms: float = Field(
        ...,
        description="Total round-trip pipeline execution latency in milliseconds.",
    )
    fallback_triggered: bool = Field(
        default=False,
        description="Indicates whether retrieval had no matching vectors and LLM operated with fallback instructions.",
    )
