"""Request and response schemas for the legal query pipeline."""

from typing import Literal, Optional
from pydantic import BaseModel, Field

from app.core.constants import (
    MAX_QUERY_CHARACTER_LENGTH,
    MIN_QUERY_CHARACTER_LENGTH,
    VALID_LEGAL_DOMAINS,
)
from app.rag.claim_models import ClaimVerificationSummary

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
    similarity_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score for primary vector hits; None for supporting neighbor chunks.",
    )
    # Extended Evidence Metadata
    document_id: Optional[str] = Field(
        default=None,
        description="Originating document ID if retrieved from an ingested file.",
    )
    document_name: Optional[str] = Field(
        default=None,
        description="Filename of the ingested document.",
    )
    page_number: Optional[int] = Field(
        default=None,
        description="Page number in the original PDF/DOCX where this provision appears.",
    )
    heading: Optional[str] = Field(
        default=None,
        description="Heading or title extracted from the document structure.",
    )
    chunk_id: Optional[str] = Field(
        default=None,
        description="Deterministic chunk identifier from vector index.",
    )
    evidence_role: Optional[str] = Field(
        default="primary",
        description="Role of evidence in research graph: 'primary' (direct vector hit) or 'supporting' (neighboring expansion).",
    )
    exhibit_id: Optional[str] = Field(
        default=None,
        description="Assigned context exhibit identifier (e.g., 'EXHIBIT_1').",
    )
    support_status: Optional[str] = Field(
        default="SUPPORTED",
        description="Empirical evidence support status: 'SUPPORTED', 'PARTIALLY_SUPPORTED', or 'UNSUPPORTED'.",
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
    document_id: Optional[str] = Field(
        default=None,
        description="Optional document ID to scope retrieval to a specific ingested document.",
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
    detected_domain: Optional[str] = Field(
        default=None,
        description="Legal domain detected during classification or preserved from client request.",
    )
    filter_relaxed: bool = Field(
        default=False,
        description="Indicates whether an auto-inferred domain filter was relaxed to unconstrained search.",
    )
    expansion_applied: bool = Field(
        default=False,
        description="Indicates whether structural neighboring chunk expansion was applied.",
    )
    expansion_count: int = Field(
        default=0,
        description="Total neighboring supporting chunks added to context.",
    )
    verification: Optional[ClaimVerificationSummary] = Field(
        default=None,
        description="Empirical citation verification and claim attribution audit.",
    )
