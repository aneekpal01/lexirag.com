"""Data models for multi-document research, evidence grouping, and cross-document relations."""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field

from app.rag.claim_models import SupportStatus


class ComparisonRelationType(str, Enum):
    """
    Categorization of relational semantics between provisions in distinct documents.
    
    IMPORTANT: This indicates empirical textual relationship between retrieved provisions.
    It does NOT assert universal judicial resolution or binding legal contradiction.
    """

    DIFFERS = "DIFFERS"
    # Provisions address the same subject matter but establish divergent requirements, thresholds, or terms.

    SUPPORTS = "SUPPORTS"
    # Provisions are mutually consistent, reinforce each other, or mirror identical obligations.

    OVERLAPS = "OVERLAPS"
    # Provisions share subject matter, but one document introduces specific conditions or exceptions.

    POTENTIAL_CONFLICT = "POTENTIAL_CONFLICT"
    # Provisions impose mutually incompatible rules or contradictory prohibitions requiring legal reconciliation.

    MENTIONS = "MENTIONS"
    # An ancillary mention of the topic without establishing detailed operational mechanisms.


class ComparisonRelation(BaseModel):
    """Structured relationship between provisions in two compared documents."""

    relation_id: str = Field(..., description="Deterministic relation identifier (e.g., 'rel-1').")
    source_document_id: str = Field(..., description="Canonical ID of the source document.")
    source_document_name: str = Field(..., description="Filename or label of the source document.")
    source_provision: str = Field(..., description="Section or clause in source document (e.g. 'Section 14').")
    target_document_id: str = Field(..., description="Canonical ID of the target document.")
    target_document_name: str = Field(..., description="Filename or label of the target document.")
    target_provision: str = Field(..., description="Section or clause in target document (e.g. 'Section 14').")
    relation_type: ComparisonRelationType = Field(..., description="Classified relational category.")
    provision_topic: Optional[str] = Field(default=None, description="Subject matter or provision topic (e.g., 'termination', 'confidentiality').")
    claim_id: Optional[str] = Field(default=None, description="Associated claim ID from citation verification.")
    cited_exhibits: list[str] = Field(
        default_factory=list,
        description="Exhibit IDs grounding this comparison across both documents.",
    )
    summary: str = Field(..., description="Cautious natural language explanation of the apparent difference or relation.")
    support_status: SupportStatus = Field(
        default=SupportStatus.SUPPORTED,
        description="Verification outcome of the comparative assertion against indexed evidence.",
    )
    is_potential_conflict: bool = Field(
        default=False,
        description="True if provisions present potential operational or legal friction.",
    )

    @property
    def doc_a_id(self) -> str:
        return self.source_document_id

    @property
    def doc_b_id(self) -> str:
        return self.target_document_id

    @property
    def doc_a_name(self) -> str:
        return self.source_document_name

    @property
    def doc_b_name(self) -> str:
        return self.target_document_name

    @property
    def rationale(self) -> str:
        return self.summary

    def to_dict(self) -> dict[str, Any]:
        """Converts relation to standard dictionary for serialization."""
        return {
            "relation_id": self.relation_id,
            "source_document_id": self.source_document_id,
            "source_document_name": self.source_document_name,
            "source_provision": self.source_provision,
            "target_document_id": self.target_document_id,
            "target_document_name": self.target_document_name,
            "target_provision": self.target_provision,
            "relation_type": self.relation_type.value,
            "provision_topic": self.provision_topic,
            "claim_id": self.claim_id,
            "cited_exhibits": self.cited_exhibits,
            "summary": self.summary,
            "rationale": self.summary,
            "support_status": self.support_status.value,
            "is_potential_conflict": self.is_potential_conflict,
            "doc_a_id": self.source_document_id,
            "doc_b_id": self.target_document_id,
            "doc_a_name": self.source_document_name,
            "doc_b_name": self.target_document_name,
        }


class DocumentEvidenceGroup(BaseModel):
    """Scoped summary of evidence retrieved for a single target document."""

    document_id: str = Field(..., description="Canonical SHA-256 identifier of the document.")
    document_name: str = Field(..., description="Sanitized document filename.")
    primary_chunk_count: int = Field(default=0, ge=0, description="Total direct vector hit chunks retained.")
    supporting_chunk_count: int = Field(default=0, ge=0, description="Total expanded neighboring chunks retained.")
    exhibit_ids: list[str] = Field(default_factory=list, description="Assigned exhibit labels for this document.")
    provisions_covered: list[str] = Field(default_factory=list, description="Sections or headings found in this document.")
    has_sufficient_evidence: bool = Field(
        default=True,
        description="False if the indexed corpus yielded zero matching provisions for this document.",
    )

    def to_dict(self) -> dict[str, Any]:
        """Converts group to standard dictionary for serialization."""
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "primary_chunk_count": self.primary_chunk_count,
            "supporting_chunk_count": self.supporting_chunk_count,
            "exhibit_ids": self.exhibit_ids,
            "provisions_covered": self.provisions_covered,
            "has_sufficient_evidence": self.has_sufficient_evidence,
        }


class ComparisonAnalysisSummary(BaseModel):
    """Public schema summarizing multi-document research, provision comparison, and relation audit."""

    is_comparison: bool = Field(default=True, description="True if multi-document comparison mode was executed.")
    is_multi_document: bool = Field(default=True, description="True if multiple documents participated.")
    document_count: int = Field(default=0, ge=0, description="Number of participating documents.")
    target_document_ids: list[str] = Field(default_factory=list, description="Explicit document IDs requested.")
    document_groups: list[DocumentEvidenceGroup] = Field(
        default_factory=list,
        description="Per-document evidence breakdown.",
    )
    relations: list[ComparisonRelation] = Field(
        default_factory=list,
        description="Identified cross-document relationships across provisions.",
    )
    missing_documents: list[str] = Field(
        default_factory=list,
        description="Requested document IDs that yielded zero sufficient evidence in the indexed corpus.",
    )
    comparison_notes: list[str] = Field(
        default_factory=list,
        description="Evidentiary and procedural caveats regarding the comparison.",
    )

    def __len__(self) -> int:
        return len(self.relations)

    def __getitem__(self, index: int) -> ComparisonRelation:
        return self.relations[index]

    def __iter__(self):
        return iter(self.relations)

    def to_dict(self) -> dict[str, Any]:
        """Converts summary to standard dictionary for serialization."""
        return {
            "is_comparison": self.is_comparison,
            "is_multi_document": self.is_multi_document,
            "document_count": self.document_count or len(self.document_groups),
            "target_document_ids": self.target_document_ids,
            "document_groups": [g.to_dict() for g in self.document_groups],
            "relations": [r.to_dict() for r in self.relations],
            "missing_documents": self.missing_documents,
            "comparison_notes": self.comparison_notes,
        }
