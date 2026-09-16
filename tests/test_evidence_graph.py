"""Comprehensive tests for LexiRAG Phase 3: Evidence Graph & Controlled Context Expansion."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_nebius_client, get_qdrant_wrapper
from app.clients.nebius import NebiusTokenFactoryClient
from app.clients.qdrant import (
    MalformedQdrantPayloadError,
    QdrantClientWrapper,
    RetrievedStatutoryChunk,
)
from app.core.config import Settings
from app.core.constants import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_EXPANSION_DEPTH,
    DEFAULT_MAX_NEIGHBORS_PER_CHUNK,
    DEFAULT_MAX_PRIMARY_CHUNKS_TO_EXPAND,
    DEFAULT_MAX_TOTAL_EXPANDED_CHUNKS,
)
from app.core.exceptions import QdrantServiceError
from app.main import create_application
from app.rag.context import (
    build_grounded_context_block,
    deduplicate_evidence_chunks,
)
from app.rag.expansion import ControlledEvidenceExpander
from app.rag.graph_models import (
    EvidenceEdge,
    EvidenceGraph,
    EvidenceNode,
    EvidenceRole,
    StructuralEdgeType,
)
from app.rag.nodes import expand_evidence_graph_node, format_citations_node
from app.rag.state import LegalGraphState
from app.schemas.query import LegalCitation, LegalQueryResponse


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def phase3_settings() -> Settings:
    return Settings(
        nebius_api_key="mock-nebius-key",
        nemotron_nano_model="nvidia/nemotron-3-nano-30b-a3b",
        nemotron_super_model="nvidia/nemotron-3-super-120b-a12b",
        embedding_model="BAAI/bge-m3",
        qdrant_url="http://localhost:6333",
        qdrant_collection_name="indian_legal_corpus",
        top_k_simple=3,
        top_k_complex=7,
        min_score_simple=0.50,
        min_score_complex=0.45,
        score_margin_ratio=0.75,
        max_context_chars=12000,
        clerk_dev_mode=True,
        enable_neighbor_expansion=True,
        max_expansion_depth=1,
        max_neighbors_per_chunk=2,
        max_primary_chunks_to_expand=2,
        max_total_expanded_chunks=3,
    )


@pytest.fixture
def mock_qdrant_wrapper() -> QdrantClientWrapper:
    wrapper = MagicMock(spec=QdrantClientWrapper)
    wrapper.get_chunks_by_ids = AsyncMock(return_value=[])
    wrapper.check_health = AsyncMock(return_value=True)
    return wrapper


def create_sample_chunk_dict(
    chunk_id: str,
    doc_id: str = "doc-123",
    doc_name: str = "companies_act.pdf",
    act_name: str = "Companies Act, 2013",
    section: str = "Section 185",
    sub_section: str = "(1)",
    heading: str = "Loans to Directors",
    content: str = "No company shall advance loans to directors...",
    similarity_score: float | None = 0.88,
    evidence_role: str = "primary",
    chunk_index: int = 10,
    prev_chunk_id: str | None = None,
    next_chunk_id: str | None = None,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "document_id": doc_id,
        "document_name": doc_name,
        "act_name": act_name,
        "section": section,
        "sub_section": sub_section,
        "heading": heading,
        "page_number": 45,
        "content": content,
        "similarity_score": similarity_score,
        "evidence_role": evidence_role,
        "chunk_index": chunk_index,
        "prev_chunk_id": prev_chunk_id,
        "next_chunk_id": next_chunk_id,
        "domain": "corporate_law",
    }


def create_retrieved_chunk(
    chunk_id: str,
    doc_id: str = "doc-123",
    doc_name: str = "companies_act.pdf",
    act_name: str = "Companies Act, 2013",
    section: str = "Section 185",
    sub_section: str = "(1)",
    heading: str = "Loans to Directors",
    content: str = "No company shall advance loans to directors...",
    chunk_index: int = 10,
    prev_chunk_id: str | None = None,
    next_chunk_id: str | None = None,
) -> RetrievedStatutoryChunk:
    return RetrievedStatutoryChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        document_name=doc_name,
        act_name=act_name,
        section=section,
        sub_section=sub_section,
        heading=heading,
        page_number=45,
        content=content,
        similarity_score=None,  # Supporting chunks have None
        chunk_index=chunk_index,
        prev_chunk_id=prev_chunk_id,
        next_chunk_id=next_chunk_id,
        domain="corporate_law",
    )


# ==============================================================================
# 1. Qdrant Direct Point Retrieval Unit Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_qdrant_get_chunks_by_ids_empty(phase3_settings):
    """Empty list returns immediately without querying client."""
    mock_inner_client = AsyncMock()
    wrapper = QdrantClientWrapper(client=mock_inner_client, settings=phase3_settings)

    result = await wrapper.get_chunks_by_ids([])
    assert result == []
    mock_inner_client.retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_qdrant_get_chunks_by_ids_success(phase3_settings):
    """Retrieves valid points by ID without vector search and sets score=None."""
    mock_inner_client = AsyncMock()
    wrapper = QdrantClientWrapper(client=mock_inner_client, settings=phase3_settings)

    mock_point = MagicMock()
    mock_point.id = "chunk-neighbor-1"
    mock_point.payload = {
        "chunk_id": "chunk-neighbor-1",
        "document_id": "doc-100",
        "document_name": "constitution.pdf",
        "act_name": "Constitution of India",
        "section": "Article 21",
        "content": "Protection of life and personal liberty...",
        "chunk_index": 21,
        "prev_chunk_id": "chunk-neighbor-0",
        "next_chunk_id": "chunk-neighbor-2",
    }
    mock_inner_client.retrieve.return_value = [mock_point]

    chunks = await wrapper.get_chunks_by_ids(["chunk-neighbor-1"])

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "chunk-neighbor-1"
    assert chunks[0].similarity_score is None
    assert chunks[0].prev_chunk_id == "chunk-neighbor-0"
    assert chunks[0].next_chunk_id == "chunk-neighbor-2"
    mock_inner_client.retrieve.assert_awaited_once_with(
        collection_name="indian_legal_corpus",
        ids=["chunk-neighbor-1"],
        with_payload=True,
        with_vectors=False,
    )


@pytest.mark.asyncio
async def test_qdrant_get_chunks_by_ids_handles_missing_and_malformed(phase3_settings):
    """Skips malformed records and handles missing points without crashing."""
    mock_inner_client = AsyncMock()
    wrapper = QdrantClientWrapper(client=mock_inner_client, settings=phase3_settings)

    good_point = MagicMock()
    good_point.id = "chunk-good"
    good_point.payload = {
        "chunk_id": "chunk-good",
        "document_id": "doc-100",
        "act_name": "Companies Act, 2013",
        "section": "Section 1",
        "content": "Short title and commencement...",
    }

    bad_point = MagicMock()
    bad_point.id = "chunk-bad"
    bad_point.payload = {}  # missing mandatory 'act_name' or 'content'

    mock_inner_client.retrieve.return_value = [good_point, bad_point]

    chunks = await wrapper.get_chunks_by_ids(["chunk-good", "chunk-bad", "chunk-missing"])
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "chunk-good"


@pytest.mark.asyncio
async def test_qdrant_get_chunks_by_ids_raises_qdrant_service_error(phase3_settings):
    """Wraps network/connection errors into QdrantServiceError."""
    mock_inner_client = AsyncMock()
    mock_inner_client.retrieve.side_effect = RuntimeError("Qdrant socket closed unexpectedly")
    wrapper = QdrantClientWrapper(client=mock_inner_client, settings=phase3_settings)

    with pytest.raises(QdrantServiceError) as exc_info:
        await wrapper.get_chunks_by_ids(["chunk-err"])

    assert "Point lookup by ID failed" in str(exc_info.value)


# ==============================================================================
# 2. Controlled Evidence Expander Unit Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_expander_disabled_returns_only_primary(mock_qdrant_wrapper, phase3_settings):
    """When enable_neighbor_expansion=False, returns only primary nodes without querying neighbors."""
    phase3_settings.enable_neighbor_expansion = False
    expander = ControlledEvidenceExpander(qdrant_wrapper=mock_qdrant_wrapper, settings=phase3_settings)

    primary = create_sample_chunk_dict(
        "chunk-1", prev_chunk_id="chunk-0", next_chunk_id="chunk-2"
    )
    graph = await expander.expand_evidence([primary])

    assert len(graph.nodes) == 1
    assert len(graph.get_primary_nodes()) == 1
    assert len(graph.get_supporting_nodes()) == 0
    assert len(graph.edges) == 0
    mock_qdrant_wrapper.get_chunks_by_ids.assert_not_called()


@pytest.mark.asyncio
async def test_expander_empty_input_returns_empty_graph(mock_qdrant_wrapper, phase3_settings):
    """Empty primary chunks returns empty graph cleanly."""
    expander = ControlledEvidenceExpander(qdrant_wrapper=mock_qdrant_wrapper, settings=phase3_settings)
    graph = await expander.expand_evidence([])

    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0
    mock_qdrant_wrapper.get_chunks_by_ids.assert_not_called()


@pytest.mark.asyncio
async def test_expander_single_prev_expansion(mock_qdrant_wrapper, phase3_settings):
    """Expands previous neighbor chunk with role SUPPORTING_PREV and PREVIOUS edge."""
    expander = ControlledEvidenceExpander(qdrant_wrapper=mock_qdrant_wrapper, settings=phase3_settings)

    primary = create_sample_chunk_dict("chunk-2", prev_chunk_id="chunk-1", next_chunk_id=None)
    neighbor_chunk = create_retrieved_chunk(
        chunk_id="chunk-1",
        section="Section 184",
        heading="Disclosure of Interest",
        content="Every director shall disclose his interest...",
        chunk_index=9,
    )
    mock_qdrant_wrapper.get_chunks_by_ids.return_value = [neighbor_chunk]

    graph = await expander.expand_evidence([primary])

    assert len(graph.nodes) == 2
    assert len(graph.get_primary_nodes()) == 1
    assert len(graph.get_supporting_nodes()) == 1

    supp_node = graph.nodes["chunk-1"]
    assert supp_node.evidence_role == EvidenceRole.SUPPORTING_PREV
    assert supp_node.similarity_score is None
    assert supp_node.parent_chunk_id == "chunk-2"

    # Edge from parent to neighbor
    prev_edges = [e for e in graph.edges if e.edge_type == StructuralEdgeType.PREVIOUS]
    assert len(prev_edges) == 1
    assert prev_edges[0].source_chunk_id == "chunk-2"
    assert prev_edges[0].target_chunk_id == "chunk-1"


@pytest.mark.asyncio
async def test_expander_single_next_expansion(mock_qdrant_wrapper, phase3_settings):
    """Expands next neighbor chunk with role SUPPORTING_NEXT and NEXT edge."""
    expander = ControlledEvidenceExpander(qdrant_wrapper=mock_qdrant_wrapper, settings=phase3_settings)

    primary = create_sample_chunk_dict("chunk-2", prev_chunk_id=None, next_chunk_id="chunk-3")
    neighbor_chunk = create_retrieved_chunk(
        chunk_id="chunk-3",
        section="Section 186",
        heading="Loan and Investment by Company",
        content="No company shall directly or indirectly make loan...",
        chunk_index=11,
    )
    mock_qdrant_wrapper.get_chunks_by_ids.return_value = [neighbor_chunk]

    graph = await expander.expand_evidence([primary])

    assert len(graph.nodes) == 2
    supp_node = graph.nodes["chunk-3"]
    assert supp_node.evidence_role == EvidenceRole.SUPPORTING_NEXT
    assert supp_node.similarity_score is None

    next_edges = [e for e in graph.edges if e.edge_type == StructuralEdgeType.NEXT]
    assert len(next_edges) == 1
    assert next_edges[0].source_chunk_id == "chunk-2"
    assert next_edges[0].target_chunk_id == "chunk-3"


@pytest.mark.asyncio
async def test_expander_bidirectional_expansion(mock_qdrant_wrapper, phase3_settings):
    """Expands both prev and next neighbors for a single primary chunk."""
    expander = ControlledEvidenceExpander(qdrant_wrapper=mock_qdrant_wrapper, settings=phase3_settings)

    primary = create_sample_chunk_dict("chunk-5", prev_chunk_id="chunk-4", next_chunk_id="chunk-6")
    prev_chunk = create_retrieved_chunk("chunk-4", chunk_index=4)
    next_chunk = create_retrieved_chunk("chunk-6", chunk_index=6)
    mock_qdrant_wrapper.get_chunks_by_ids.return_value = [prev_chunk, next_chunk]

    graph = await expander.expand_evidence([primary])

    assert len(graph.nodes) == 3
    assert "chunk-4" in graph.nodes
    assert "chunk-6" in graph.nodes
    assert graph.nodes["chunk-4"].evidence_role == EvidenceRole.SUPPORTING_PREV
    assert graph.nodes["chunk-6"].evidence_role == EvidenceRole.SUPPORTING_NEXT
    mock_qdrant_wrapper.get_chunks_by_ids.assert_awaited_once_with(["chunk-4", "chunk-6"])


@pytest.mark.asyncio
async def test_expander_neighbor_already_in_primary_graph(mock_qdrant_wrapper, phase3_settings):
    """If candidate neighbor chunk is already in graph as a primary chunk, do not fetch; link edge."""
    expander = ControlledEvidenceExpander(qdrant_wrapper=mock_qdrant_wrapper, settings=phase3_settings)

    chunk_a = create_sample_chunk_dict("chunk-10", prev_chunk_id=None, next_chunk_id="chunk-11")
    chunk_b = create_sample_chunk_dict("chunk-11", prev_chunk_id="chunk-10", next_chunk_id=None)

    graph = await expander.expand_evidence([chunk_a, chunk_b])

    # No IDs needed to be fetched from Qdrant because chunk-11 is already in graph
    mock_qdrant_wrapper.get_chunks_by_ids.assert_not_called()
    assert len(graph.nodes) == 2
    assert len(graph.get_primary_nodes()) == 2
    assert len(graph.get_supporting_nodes()) == 0

    # NEXT edge from chunk-10 to chunk-11
    next_edges = [e for e in graph.edges if e.edge_type == StructuralEdgeType.NEXT]
    assert len(next_edges) == 1
    assert next_edges[0].source_chunk_id == "chunk-10"
    assert next_edges[0].target_chunk_id == "chunk-11"


@pytest.mark.asyncio
async def test_expander_caps_primary_chunks_to_expand(mock_qdrant_wrapper, phase3_settings):
    """Caps expansion to first max_primary_chunks_to_expand (default 2), ignoring later candidates."""
    phase3_settings.max_primary_chunks_to_expand = 2
    expander = ControlledEvidenceExpander(qdrant_wrapper=mock_qdrant_wrapper, settings=phase3_settings)

    chunk1 = create_sample_chunk_dict("c1", prev_chunk_id="c0", next_chunk_id=None)
    chunk2 = create_sample_chunk_dict("c2", prev_chunk_id="c1.5", next_chunk_id=None)
    chunk3 = create_sample_chunk_dict("c3", prev_chunk_id="c2.5", next_chunk_id=None)

    mock_qdrant_wrapper.get_chunks_by_ids.return_value = [
        create_retrieved_chunk("c0"),
        create_retrieved_chunk("c1.5"),
    ]

    graph = await expander.expand_evidence([chunk1, chunk2, chunk3])

    # Should only lookup c0 and c1.5; c2.5 from 3rd primary must be ignored
    called_ids = mock_qdrant_wrapper.get_chunks_by_ids.call_args[0][0]
    assert "c0" in called_ids
    assert "c1.5" in called_ids
    assert "c2.5" not in called_ids


@pytest.mark.asyncio
async def test_expander_caps_total_expanded_chunks(mock_qdrant_wrapper, phase3_settings):
    """Caps total expanded supporting chunks to max_total_expanded_chunks (default 3)."""
    phase3_settings.max_primary_chunks_to_expand = 2
    phase3_settings.max_total_expanded_chunks = 2  # limit to 2 total
    expander = ControlledEvidenceExpander(qdrant_wrapper=mock_qdrant_wrapper, settings=phase3_settings)

    chunk1 = create_sample_chunk_dict("c1", prev_chunk_id="c1-prev", next_chunk_id="c1-next")
    chunk2 = create_sample_chunk_dict("c2", prev_chunk_id="c2-prev", next_chunk_id="c2-next")

    mock_qdrant_wrapper.get_chunks_by_ids.return_value = [
        create_retrieved_chunk("c1-prev"),
        create_retrieved_chunk("c1-next"),
    ]

    await expander.expand_evidence([chunk1, chunk2])

    called_ids = mock_qdrant_wrapper.get_chunks_by_ids.call_args[0][0]
    assert len(called_ids) == 2
    assert "c1-prev" in called_ids
    assert "c1-next" in called_ids
    assert "c2-prev" not in called_ids


@pytest.mark.asyncio
async def test_expander_bounded_depth_one_no_recursion(mock_qdrant_wrapper, phase3_settings):
    """Neighbors themselves may contain pointers, but expansion stops at depth 1 without chaining."""
    expander = ControlledEvidenceExpander(qdrant_wrapper=mock_qdrant_wrapper, settings=phase3_settings)

    primary = create_sample_chunk_dict("p1", prev_chunk_id="n1", next_chunk_id=None)
    # Neighbor n1 itself points to n0, but n0 must NOT be fetched
    neighbor_n1 = create_retrieved_chunk("n1", prev_chunk_id="n0", next_chunk_id="p1")
    mock_qdrant_wrapper.get_chunks_by_ids.return_value = [neighbor_n1]

    graph = await expander.expand_evidence([primary])

    assert len(graph.nodes) == 2
    assert "n1" in graph.nodes
    assert "n0" not in graph.nodes
    mock_qdrant_wrapper.get_chunks_by_ids.assert_awaited_once_with(["n1"])


# ==============================================================================
# 3. Security & Document Isolation Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_document_isolation_rejects_cross_document_neighbor(mock_qdrant_wrapper, phase3_settings):
    """Rejects neighbor chunk if its document_id does not match the parent primary chunk."""
    expander = ControlledEvidenceExpander(qdrant_wrapper=mock_qdrant_wrapper, settings=phase3_settings)

    primary = create_sample_chunk_dict(
        "primary-1", doc_id="doc-company-act", prev_chunk_id="foreign-chunk-9"
    )
    # Neighbor belongs to a different document!
    rogue_neighbor = create_retrieved_chunk(
        "foreign-chunk-9", doc_id="doc-malicious-leak", act_name="Rogue Doc"
    )
    mock_qdrant_wrapper.get_chunks_by_ids.return_value = [rogue_neighbor]

    graph = await expander.expand_evidence([primary])

    # Foreign chunk must be strictly dropped
    assert "foreign-chunk-9" not in graph.nodes
    assert len(graph.nodes) == 1
    assert len(graph.edges) == 0


@pytest.mark.asyncio
async def test_document_isolation_rejects_missing_document_id(mock_qdrant_wrapper, phase3_settings):
    """Rejects neighbor chunk if its document_id is missing/None."""
    expander = ControlledEvidenceExpander(qdrant_wrapper=mock_qdrant_wrapper, settings=phase3_settings)

    primary = create_sample_chunk_dict("p1", doc_id="doc-legit", prev_chunk_id="n1")
    rogue_neighbor = create_retrieved_chunk("n1", doc_id=None)
    mock_qdrant_wrapper.get_chunks_by_ids.return_value = [rogue_neighbor]

    graph = await expander.expand_evidence([primary])

    assert "n1" not in graph.nodes
    assert len(graph.nodes) == 1


@pytest.mark.asyncio
async def test_document_isolation_enforces_explicit_query_document_id(mock_qdrant_wrapper, phase3_settings):
    """If user specified explicit_document_id, rejects neighbor if it doesn't match the query filter."""
    expander = ControlledEvidenceExpander(qdrant_wrapper=mock_qdrant_wrapper, settings=phase3_settings)

    primary = create_sample_chunk_dict("p1", doc_id="doc-other", prev_chunk_id="n1")
    neighbor = create_retrieved_chunk("n1", doc_id="doc-other")
    mock_qdrant_wrapper.get_chunks_by_ids.return_value = [neighbor]

    # Explicit filter requires doc-specific
    graph = await expander.expand_evidence([primary], explicit_document_id="doc-specific")

    assert "n1" not in graph.nodes


# ==============================================================================
# 4. Score Rule & Structural Edge Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_score_rule_supporting_chunks_have_none_score(mock_qdrant_wrapper, phase3_settings):
    """Mandatory Correction #2: Supporting chunks have similarity_score=None, not 0.0 or copied score."""
    expander = ControlledEvidenceExpander(qdrant_wrapper=mock_qdrant_wrapper, settings=phase3_settings)

    primary = create_sample_chunk_dict("p1", similarity_score=0.8765, next_chunk_id="s1")
    neighbor = create_retrieved_chunk("s1")
    mock_qdrant_wrapper.get_chunks_by_ids.return_value = [neighbor]

    graph = await expander.expand_evidence([primary])

    assert graph.nodes["p1"].similarity_score == 0.8765
    assert graph.nodes["s1"].similarity_score is None


@pytest.mark.asyncio
async def test_structural_same_section_edges(mock_qdrant_wrapper, phase3_settings):
    """Adds SAME_SECTION edge between chunks sharing identical Act and Section."""
    expander = ControlledEvidenceExpander(qdrant_wrapper=mock_qdrant_wrapper, settings=phase3_settings)

    chunk_a = create_sample_chunk_dict(
        "cA", act_name="Companies Act, 2013", section="Section 185", sub_section="(1)"
    )
    chunk_b = create_sample_chunk_dict(
        "cB", act_name="Companies Act, 2013", section="Section 185", sub_section="(2)"
    )

    graph = await expander.expand_evidence([chunk_a, chunk_b])

    same_sec_edges = [e for e in graph.edges if e.edge_type == StructuralEdgeType.SAME_SECTION]
    assert len(same_sec_edges) == 1
    assert same_sec_edges[0].source_chunk_id == "cA"
    assert same_sec_edges[0].target_chunk_id == "cB"


@pytest.mark.asyncio
async def test_graceful_degradation_on_qdrant_error(mock_qdrant_wrapper, phase3_settings):
    """Gracefully falls back to primary nodes if Qdrant get_chunks_by_ids raises error."""
    mock_qdrant_wrapper.get_chunks_by_ids.side_effect = RuntimeError("Qdrant transient timeout")
    expander = ControlledEvidenceExpander(qdrant_wrapper=mock_qdrant_wrapper, settings=phase3_settings)

    primary = create_sample_chunk_dict("p1", prev_chunk_id="n1")
    graph = await expander.expand_evidence([primary])

    # No crash, returns primary nodes intact
    assert len(graph.nodes) == 1
    assert "p1" in graph.nodes
    assert len(graph.get_supporting_nodes()) == 0


# ==============================================================================
# 5. Context Budgeting & Deduplication Priority Tests
# ==============================================================================


def test_context_priority_budgeting_truncates_supporting_first():
    """Primary evidence receives guaranteed priority; supporting neighbor context is truncated first."""
    large_primary = create_sample_chunk_dict(
        "p1",
        section="Section 185",
        content="Primary substantive statute text with important details " * 20,
        evidence_role="primary",
    )
    large_supporting = create_sample_chunk_dict(
        "s1",
        section="Section 184",
        content="Supporting context text with extra background info " * 20,
        evidence_role="supporting_prev",
    )

    # Budget big enough for 1 exhibit block (~1000 chars) but not both (~2000 chars)
    max_budget = 1200
    context_str, retained = build_grounded_context_block(
        [large_supporting, large_primary],  # passed in random order
        max_chars=max_budget,
    )

    # Primary should be kept, supporting should be truncated
    assert len(retained) == 1
    assert retained[0]["chunk_id"] == "p1"
    assert "Role: Primary Evidence" in context_str
    assert "Role: Supporting Context" not in context_str


def test_context_exhibit_role_headers_and_scores():
    """Checks human-readable role and score strings in exhibit blocks."""
    primary = create_sample_chunk_dict(
        "p1",
        act_name="Companies Act, 2013",
        section="Section 185",
        similarity_score=0.8912,
        evidence_role="primary",
    )
    supporting_prev = create_sample_chunk_dict(
        "s1",
        act_name="Companies Act, 2013",
        section="Section 184",
        similarity_score=None,
        evidence_role="supporting_prev",
    )

    context_str, _ = build_grounded_context_block([primary, supporting_prev])

    assert "[EXHIBIT 1] Source: Companies Act, 2013 | Provision: Section 185" in context_str
    assert "Role: Primary Evidence" in context_str
    assert "Match Score: 0.8912" in context_str

    assert "[EXHIBIT 2] Source: Companies Act, 2013 | Provision: Section 184" in context_str
    assert "Role: Supporting Context (Sequential Predecessor)" in context_str
    assert "Match Score: Structural Context (No Vector Score)" in context_str


def test_deduplication_prefers_primary_role_and_valid_score():
    """When a chunk is listed both as supporting and primary, primary role and score are preserved."""
    chunk_supp = create_sample_chunk_dict("dup-1", similarity_score=None, evidence_role="supporting_prev")
    chunk_prim = create_sample_chunk_dict("dup-1", similarity_score=0.91, evidence_role="primary")

    deduped = deduplicate_evidence_chunks([chunk_supp, chunk_prim])

    assert len(deduped) == 1
    assert deduped[0]["evidence_role"] == "primary"
    assert deduped[0]["similarity_score"] == 0.91


# ==============================================================================
# 6. RAG Node Execution & Citation Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_expand_evidence_graph_node_bypasses_on_empty(mock_qdrant_wrapper, phase3_settings):
    """When filtered_chunks is empty or fallback_triggered is True, bypasses expansion."""
    state: LegalGraphState = {
        "query": "Test query",
        "filtered_chunks": [],
        "fallback_triggered": False,
    }

    res = await expand_evidence_graph_node(state, mock_qdrant_wrapper, phase3_settings)

    assert res["expanded_chunks"] == []
    assert res["evidence_graph"] is None
    assert res["expansion_applied"] is False
    assert res["expansion_count"] == 0
    mock_qdrant_wrapper.get_chunks_by_ids.assert_not_called()


@pytest.mark.asyncio
async def test_expand_evidence_graph_node_success(mock_qdrant_wrapper, phase3_settings):
    """Executes node, merges primary + expanded into filtered_chunks, provides diagnostics."""
    primary = create_sample_chunk_dict("p1", prev_chunk_id="p0")
    neighbor = create_retrieved_chunk("p0")
    mock_qdrant_wrapper.get_chunks_by_ids.return_value = [neighbor]

    state: LegalGraphState = {
        "query": "Test query",
        "filtered_chunks": [primary],
        "fallback_triggered": False,
    }

    res = await expand_evidence_graph_node(state, mock_qdrant_wrapper, phase3_settings)

    assert res["expansion_applied"] is True
    assert res["expansion_count"] == 1
    assert len(res["filtered_chunks"]) == 2
    assert res["evidence_graph"] is not None
    assert res["evidence_graph"]["primary_count"] == 1
    assert res["evidence_graph"]["supporting_count"] == 1


@pytest.mark.asyncio
async def test_format_citations_node_supports_none_scores_and_roles():
    """Format citations handles supporting chunks with similarity_score=None and distinct roles."""
    state: LegalGraphState = {
        "query": "Test query",
        "filtered_chunks": [
            create_sample_chunk_dict("p1", section="Section 185", similarity_score=0.88, evidence_role="primary"),
            create_sample_chunk_dict("s1", section="Section 184", similarity_score=None, evidence_role="supporting_prev"),
        ],
        "generated_answer": "According to Section 185, loans are restricted.",
    }

    res = await format_citations_node(state)
    citations = res["citations"]

    assert len(citations) == 2
    assert citations[0]["similarity_score"] == 0.88
    assert citations[0]["evidence_role"] == "primary"
    assert citations[1]["similarity_score"] is None
    assert citations[1]["evidence_role"] == "supporting_prev"


# ==============================================================================
# 7. FastAPI Integration Test with Expansion Metadata
# ==============================================================================


def test_api_query_returns_expansion_metadata():
    """FastAPI query endpoint returns expansion_applied and expansion_count fields."""
    app = create_application()

    mock_nebius = MagicMock(spec=NebiusTokenFactoryClient)
    mock_nebius.create_chat_completion = AsyncMock(
        side_effect=[
            "CLASSIFICATION: SIMPLE\nRATIONALE: Basic statutory compliance.",
            "Under Section 185 of the Companies Act 2013...",
        ]
    )
    mock_nebius.create_embedding = AsyncMock(return_value=[0.01] * 1024)

    mock_qdrant = MagicMock(spec=QdrantClientWrapper)
    primary_chunk = RetrievedStatutoryChunk(
        chunk_id="primary-1",
        act_name="Companies Act, 2013",
        section="Section 185",
        sub_section="(1)",
        title="Loans to Directors",
        content="No company shall advance loans to directors...",
        domain="corporate_law",
        similarity_score=0.91,
        document_id="doc-123",
        prev_chunk_id="neighbor-0",
    )
    neighbor_chunk = RetrievedStatutoryChunk(
        chunk_id="neighbor-0",
        act_name="Companies Act, 2013",
        section="Section 184",
        sub_section=None,
        title="Disclosure of Interest",
        content="Every director shall disclose...",
        domain="corporate_law",
        similarity_score=None,
        document_id="doc-123",
    )

    mock_qdrant.search_statutes = AsyncMock(return_value=[primary_chunk])
    mock_qdrant.get_chunks_by_ids = AsyncMock(return_value=[neighbor_chunk])
    mock_qdrant.check_health = AsyncMock(return_value=True)

    app.dependency_overrides[get_nebius_client] = lambda: mock_nebius
    app.dependency_overrides[get_qdrant_wrapper] = lambda: mock_qdrant

    client = TestClient(app)
    headers = {"Authorization": "Bearer dev-test-token"}
    payload = {"query": "Can a company advance loans to directors under Section 185?"}

    response = client.post("/query", json=payload, headers=headers)
    assert response.status_code == 200

    data = response.json()
    assert "expansion_applied" in data
    assert "expansion_count" in data
    assert data["expansion_applied"] is True
    assert data["expansion_count"] == 1
    assert len(data["citations"]) >= 1
