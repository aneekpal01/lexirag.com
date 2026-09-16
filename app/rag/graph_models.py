"""Data models for the LexiRAG Evidence Graph and structural context expansion."""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class EvidenceRole(str, Enum):
    """Classification of an evidence chunk's origin and role in the graph."""

    PRIMARY = "primary"                      # Directly surfaced by dense retrieval and score margin pruning
    SUPPORTING_PREV = "supporting_prev"      # Sequential predecessor to a primary chunk
    SUPPORTING_NEXT = "supporting_next"      # Sequential successor to a primary chunk


class StructuralEdgeType(str, Enum):
    """Strictly structural, metadata-derived relationships between evidence nodes."""

    PREVIOUS = "PREVIOUS"                    # target_chunk is the sequential predecessor (chunk_index - 1)
    NEXT = "NEXT"                            # target_chunk is the sequential successor (chunk_index + 1)
    SAME_SECTION = "SAME_SECTION"            # Both chunks share the same statutory section in the same Act


class EvidenceNode(BaseModel):
    """Query-local node representing a single primary or supporting evidence chunk."""

    chunk_id: str = Field(..., description="Unique deterministic chunk identifier.")
    document_id: Optional[str] = Field(default=None, description="SHA-256 hash of parent document.")
    document_name: Optional[str] = Field(default=None, description="Original filename of parent document.")
    act_name: str = Field(..., description="Act or statutory title.")
    section: str = Field(..., description="Statutory section or clause identifier.")
    sub_section: Optional[str] = Field(default=None, description="Sub-section or sub-clause.")
    heading: Optional[str] = Field(default=None, description="Structural heading.")
    page_number: Optional[int] = Field(default=None, description="Physical source page number.")
    content: str = Field(..., description="Verbatim text content of chunk.")
    similarity_score: Optional[float] = Field(
        default=None,
        description="Dense vector similarity score. Present for PRIMARY chunks, strictly None for SUPPORTING.",
    )
    evidence_role: EvidenceRole = Field(..., description="Role of chunk in evidence graph.")
    parent_chunk_id: Optional[str] = Field(
        default=None,
        description="Chunk ID of the primary node that prompted this expansion (if supporting).",
    )
    expansion_reason: Optional[str] = Field(
        default=None,
        description="Human-readable rationale for neighbor expansion.",
    )
    chunk_index: Optional[int] = Field(default=None, description="Sequential index within document.")
    prev_chunk_id: Optional[str] = Field(default=None, description="Pointer to preceding chunk ID.")
    next_chunk_id: Optional[str] = Field(default=None, description="Pointer to succeeding chunk ID.")

    def to_chunk_dict(self) -> dict[str, Any]:
        """Converts node into standard chunk dictionary for downstream RAG nodes."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "act_name": self.act_name,
            "section": self.section,
            "sub_section": self.sub_section,
            "heading": self.heading,
            "page_number": self.page_number,
            "content": self.content,
            "similarity_score": self.similarity_score,
            "evidence_role": self.evidence_role.value,
            "parent_chunk_id": self.parent_chunk_id,
            "expansion_reason": self.expansion_reason,
            "chunk_index": self.chunk_index,
            "prev_chunk_id": self.prev_chunk_id,
            "next_chunk_id": self.next_chunk_id,
        }


class EvidenceEdge(BaseModel):
    """Directed structural relationship between two evidence nodes in the query-local graph."""

    source_chunk_id: str = Field(..., description="Origin chunk ID.")
    target_chunk_id: str = Field(..., description="Destination chunk ID.")
    edge_type: StructuralEdgeType = Field(..., description="Structural relationship type.")


class EvidenceGraph(BaseModel):
    """Query-local evidence graph containing primary evidence and bounded supporting context."""

    nodes: dict[str, EvidenceNode] = Field(default_factory=dict)
    edges: list[EvidenceEdge] = Field(default_factory=list)

    def add_node(self, node: EvidenceNode) -> None:
        """Adds or updates an evidence node."""
        self.nodes[node.chunk_id] = node

    def add_edge(self, source_id: str, target_id: str, edge_type: StructuralEdgeType) -> None:
        """Adds a structural edge if both nodes exist and edge is not duplicate."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return
        for edge in self.edges:
            if edge.source_chunk_id == source_id and edge.target_chunk_id == target_id and edge.edge_type == edge_type:
                return
        self.edges.append(
            EvidenceEdge(
                source_chunk_id=source_id,
                target_chunk_id=target_id,
                edge_type=edge_type,
            )
        )

    def get_primary_nodes(self) -> list[EvidenceNode]:
        """Returns all primary evidence nodes."""
        return [n for n in self.nodes.values() if n.evidence_role == EvidenceRole.PRIMARY]

    def get_supporting_nodes(self) -> list[EvidenceNode]:
        """Returns all supporting context nodes."""
        return [n for n in self.nodes.values() if n.evidence_role != EvidenceRole.PRIMARY]

    def to_diagnostics_dict(self) -> dict[str, Any]:
        """Produces a serializable diagnostic representation of the evidence graph."""
        return {
            "total_nodes": len(self.nodes),
            "primary_count": len(self.get_primary_nodes()),
            "supporting_count": len(self.get_supporting_nodes()),
            "total_edges": len(self.edges),
            "edges": [e.model_dump() for e in self.edges],
            "node_keys": list(self.nodes.keys()),
        }
