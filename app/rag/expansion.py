"""Controlled evidence graph expansion engine for LexiRAG."""

from typing import Any, Optional
from app.clients.qdrant import QdrantClientWrapper, RetrievedStatutoryChunk
from app.core.config import Settings
from app.core.logging import get_logger
from app.rag.graph_models import (
    EvidenceEdge,
    EvidenceGraph,
    EvidenceNode,
    EvidenceRole,
    StructuralEdgeType,
)

logger = get_logger(__name__)


class ControlledEvidenceExpander:
    """
    Orchestrates bounded, query-local evidence graph expansion.
    
    Enforces strict document isolation, bounded neighbor expansion (depth=1),
    separate provenance for supporting chunks (similarity_score=None),
    and graceful degradation upon Qdrant errors.
    """

    def __init__(self, qdrant_wrapper: QdrantClientWrapper, settings: Settings):
        self.qdrant_wrapper = qdrant_wrapper
        self.settings = settings

    async def expand_evidence(
        self,
        primary_chunks: list[dict[str, Any]],
        explicit_document_id: Optional[str] = None,
    ) -> EvidenceGraph:
        """
        Builds a query-local EvidenceGraph starting from primary retrieval hits.
        Expands up to max_primary_chunks_to_expand primary chunks by direct neighbors (depth=1).
        """
        graph = EvidenceGraph()

        if not primary_chunks:
            return graph

        # 1. Populate primary nodes in graph
        for chunk in primary_chunks:
            raw_score = chunk.get("similarity_score")
            score_val = float(raw_score) if raw_score is not None else None

            node = EvidenceNode(
                chunk_id=chunk["chunk_id"],
                document_id=chunk.get("document_id"),
                document_name=chunk.get("document_name"),
                act_name=chunk.get("act_name") or "Unknown Statute",
                section=chunk.get("section") or "General Provision",
                sub_section=chunk.get("sub_section"),
                heading=chunk.get("heading"),
                page_number=chunk.get("page_number"),
                content=chunk.get("content", ""),
                similarity_score=score_val,
                evidence_role=EvidenceRole.PRIMARY,
                chunk_index=chunk.get("chunk_index"),
                prev_chunk_id=chunk.get("prev_chunk_id"),
                next_chunk_id=chunk.get("next_chunk_id"),
            )
            graph.add_node(node)

        # 2. Add SAME_SECTION structural edges between existing primary nodes
        primary_node_list = list(graph.nodes.values())
        for i, node_a in enumerate(primary_node_list):
            for node_b in primary_node_list[i + 1 :]:
                if (
                    node_a.act_name.strip().lower() == node_b.act_name.strip().lower()
                    and node_a.section.strip().lower() == node_b.section.strip().lower()
                ):
                    graph.add_edge(node_a.chunk_id, node_b.chunk_id, StructuralEdgeType.SAME_SECTION)

        if not self.settings.enable_neighbor_expansion:
            logger.debug("Neighbor expansion is disabled in configuration")
            return graph

        # 3. Identify candidate primary chunks for expansion (up to max_primary_chunks_to_expand)
        candidates_to_expand = primary_node_list[: self.settings.max_primary_chunks_to_expand]

        lookup_targets: dict[str, dict[str, Any]] = {}
        total_expansion_slots_remaining = self.settings.max_total_expanded_chunks

        for primary_node in candidates_to_expand:
            if total_expansion_slots_remaining <= 0:
                break

            parent_id = primary_node.chunk_id
            expected_doc_id = primary_node.document_id

            # Check previous neighbor
            prev_id = primary_node.prev_chunk_id
            if prev_id:
                if prev_id in graph.nodes:
                    # Neighbor is already in graph; link structural PREVIOUS edge without lookup
                    graph.add_edge(parent_id, prev_id, StructuralEdgeType.PREVIOUS)
                elif prev_id not in lookup_targets and total_expansion_slots_remaining > 0:
                    lookup_targets[prev_id] = {
                        "role": EvidenceRole.SUPPORTING_PREV,
                        "parent_chunk_id": parent_id,
                        "expected_document_id": expected_doc_id,
                        "edge_type": StructuralEdgeType.PREVIOUS,
                        "reason": f"Preceding provision for primary chunk {parent_id}",
                    }
                    total_expansion_slots_remaining -= 1

            # Check next neighbor
            next_id = primary_node.next_chunk_id
            if next_id:
                if next_id in graph.nodes:
                    # Neighbor is already in graph; link structural NEXT edge without lookup
                    graph.add_edge(parent_id, next_id, StructuralEdgeType.NEXT)
                elif next_id not in lookup_targets and total_expansion_slots_remaining > 0:
                    lookup_targets[next_id] = {
                        "role": EvidenceRole.SUPPORTING_NEXT,
                        "parent_chunk_id": parent_id,
                        "expected_document_id": expected_doc_id,
                        "edge_type": StructuralEdgeType.NEXT,
                        "reason": f"Succeeding provision for primary chunk {parent_id}",
                    }
                    total_expansion_slots_remaining -= 1

        if not lookup_targets:
            return graph

        # 4. Perform direct Qdrant point retrieval (avoiding vector similarity search)
        ids_to_fetch = list(lookup_targets.keys())
        try:
            fetched_chunks = await self.qdrant_wrapper.get_chunks_by_ids(ids_to_fetch)
        except Exception as exc:
            logger.warning(
                "Qdrant direct point retrieval failed for neighbor IDs %s: %s. Gracefully degrading to primary evidence.",
                ids_to_fetch,
                exc,
            )
            return graph

        # 5. Enforce Document Isolation and build supporting evidence nodes
        for neighbor in fetched_chunks:
            meta = lookup_targets.get(neighbor.chunk_id)
            if not meta:
                continue

            expected_doc = meta["expected_document_id"]
            actual_doc = neighbor.document_id

            # Security Rule: Reject neighbor if document_id is missing or mismatches parent
            if not actual_doc or actual_doc != expected_doc:
                logger.warning(
                    "Security/Isolation violation: Rejected neighbor chunk '%s'. Expected parent document_id '%s', found '%s'",
                    neighbor.chunk_id,
                    expected_doc,
                    actual_doc,
                )
                continue

            # Security Rule: If query explicitly specified document_id, verify compliance
            if explicit_document_id and actual_doc != explicit_document_id:
                logger.warning(
                    "Security/Isolation violation: Rejected neighbor chunk '%s'. Does not match explicit query document_id '%s'",
                    neighbor.chunk_id,
                    explicit_document_id,
                )
                continue

            # Create supporting node (similarity_score is strictly None per Mandatory Correction #2)
            supporting_node = EvidenceNode(
                chunk_id=neighbor.chunk_id,
                document_id=actual_doc,
                document_name=neighbor.document_name,
                act_name=neighbor.act_name,
                section=neighbor.section,
                sub_section=neighbor.sub_section,
                heading=neighbor.heading,
                page_number=neighbor.page_number,
                content=neighbor.content,
                similarity_score=None,  # Supporting chunks must not inherit or fabricate scores
                evidence_role=meta["role"],
                parent_chunk_id=meta["parent_chunk_id"],
                expansion_reason=meta["reason"],
                chunk_index=neighbor.chunk_index,
                prev_chunk_id=neighbor.prev_chunk_id,
                next_chunk_id=neighbor.next_chunk_id,
            )
            graph.add_node(supporting_node)

            # Link structural edge
            graph.add_edge(
                meta["parent_chunk_id"],
                neighbor.chunk_id,
                meta["edge_type"],
            )

            # Check for SAME_SECTION relationship with parent
            parent_node = graph.nodes.get(meta["parent_chunk_id"])
            if (
                parent_node
                and parent_node.act_name.strip().lower() == neighbor.act_name.strip().lower()
                and parent_node.section.strip().lower() == neighbor.section.strip().lower()
            ):
                graph.add_edge(parent_node.chunk_id, neighbor.chunk_id, StructuralEdgeType.SAME_SECTION)

        logger.info(
            "Evidence graph expansion completed | primary_nodes: %d | supporting_nodes: %d | edges: %d",
            len(graph.get_primary_nodes()),
            len(graph.get_supporting_nodes()),
            len(graph.edges),
        )
        return graph
