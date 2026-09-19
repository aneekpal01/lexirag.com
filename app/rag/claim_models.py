"""Data models for legal claim extraction, evidence attribution, and verification."""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class SupportStatus(str, Enum):
    """
    Empirical verification status of a generated claim against indexed corpus evidence.
    
    IMPORTANT: This indicates solely whether the claim is substantiated by the retrieved
    exhibits in the LexiRAG indexed corpus. It does NOT assert universal legal validity
    or external real-world legal truth.
    """

    SUPPORTED = "SUPPORTED"
    # The indexed evidence provides sufficient direct textual support for the substantive claim.

    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    # Only part of a multi-proposition claim is supported, or evidence supports the general
    # proposition but not all specific details.

    UNSUPPORTED = "UNSUPPORTED"
    # The indexed evidence does not provide sufficient support, or directly contradicts the claim.


class LegalClaim(BaseModel):
    """A discrete legal/factual proposition extracted from the generated answer."""

    claim_id: str = Field(..., description="Deterministic claim identifier (e.g., 'claim-1').")
    claim_text: str = Field(..., description="Verbatim text of the extracted legal assertion.")
    sentence_index: int = Field(default=0, description="Sequential sentence index in answer text.")
    cited_exhibit_ids: list[str] = Field(
        default_factory=list,
        description="Exhibit IDs explicitly referenced in the claim text (e.g., ['EXHIBIT_1']).",
    )
    attributed_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Qdrant point/chunk IDs linked to the attributed exhibits.",
    )
    support_status: SupportStatus = Field(
        default=SupportStatus.UNSUPPORTED,
        description="Verification outcome against indexed evidence.",
    )
    verification_rationale: str = Field(
        default="",
        description="Explanation of empirical support or evidentiary gap.",
    )
    supporting_exhibits: list[str] = Field(
        default_factory=list,
        description="Exhibit IDs verified to provide textual support.",
    )
    unsupported_aspects: list[str] = Field(
        default_factory=list,
        description="Specific clauses, amounts, or sections lacking evidence if partial or unsupported.",
    )
    has_conflicting_evidence: bool = Field(
        default=False,
        description="True if strong textual contradiction is detected across retrieved exhibits.",
    )
    potential_conflict: bool = Field(
        default=False,
        description="True if differing provisions suggest a potential legal conflict requiring judicial reconciliation.",
    )

    @property
    def evidentiary_gaps(self) -> list[str]:
        return self.unsupported_aspects

    is_atomic: bool = Field(
        default=True,
        description="Indicates whether the proposition was decomposed safely or kept intact.",
    )

    def to_dict(self) -> dict[str, Any]:
        """Converts claim to standard dictionary for graph state and diagnostics."""
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "sentence_index": self.sentence_index,
            "cited_exhibit_ids": self.cited_exhibit_ids,
            "attributed_chunk_ids": self.attributed_chunk_ids,
            "support_status": self.support_status.value,
            "verification_rationale": self.verification_rationale,
            "supporting_exhibits": self.supporting_exhibits,
            "unsupported_aspects": self.unsupported_aspects,
            "has_conflicting_evidence": self.has_conflicting_evidence,
            "potential_conflict": self.potential_conflict,
            "is_atomic": self.is_atomic,
        }


class ClaimVerificationSummary(BaseModel):
    """
    Public summary of citation verification and evidence attribution audit.
    
    DEFINITIONAL INTEGRITY:
    supported_claim_ratio is defined strictly as the proportion of extracted claims
    classified as supported by indexed evidence. It must NEVER be described or interpreted
    as model accuracy, legal accuracy, factual correctness, or a hallucination rate.
    """

    total_claims: int = Field(default=0, ge=0, description="Total extracted claims evaluated.")
    supported_claims: int = Field(default=0, ge=0, description="Claims directly supported by indexed evidence.")
    partially_supported_claims: int = Field(default=0, ge=0, description="Claims with partial evidentiary support.")
    unsupported_claims: int = Field(default=0, ge=0, description="Claims lacking sufficient evidence in corpus.")
    supported_claim_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Proportion of extracted claims classified as supported by indexed evidence (None if zero claims).",
    )
    unsupported_claim_texts: list[str] = Field(
        default_factory=list,
        description="Statements that lacked sufficient evidentiary support in the indexed corpus.",
    )
    has_conflicts: bool = Field(
        default=False,
        description="Indicates whether potential conflicts or contradictory provisions were detected in evidence.",
    )
    conflict_notes: list[str] = Field(
        default_factory=list,
        description="Notes detailing conflicting provisions across retrieved exhibits.",
    )
