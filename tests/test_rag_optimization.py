"""Comprehensive tests for Phase 2: Query Pipeline & RAG Optimization."""

from unittest.mock import AsyncMock, MagicMock
import pytest

from app.clients.nebius import NebiusTokenFactoryClient
from app.clients.qdrant import QdrantClientWrapper, RetrievedStatutoryChunk
from app.core.config import Settings
from app.core.constants import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MIN_SCORE_COMPLEX,
    DEFAULT_MIN_SCORE_SIMPLE,
    DEFAULT_SCORE_MARGIN_RATIO,
    DEFAULT_TOP_K_COMPLEX,
    DEFAULT_TOP_K_SIMPLE,
    LEGAL_DOMAIN_CORPORATE,
    LEGAL_DOMAIN_TAXATION,
)
from app.rag.context import (
    build_grounded_context_block,
    deduplicate_evidence_chunks,
    group_evidence_hierarchically,
    prune_chunks_by_score_margin,
)
from app.rag.nodes import (
    classify_query_node,
    format_citations_node,
    generate_answer_node,
    retrieve_context_node,
)
from app.rag.state import LegalGraphState


@pytest.fixture
def phase2_settings() -> Settings:
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
    )


@pytest.fixture
def mock_nebius_client() -> NebiusTokenFactoryClient:
    client = MagicMock(spec=NebiusTokenFactoryClient)
    client.create_chat_completion = AsyncMock(
        return_value="CLASSIFICATION: SIMPLE\nDOMAIN: corporate_law\nRATIONALE: Single section threshold."
    )
    client.create_embedding = AsyncMock(return_value=[0.05] * 1024)
    return client


@pytest.fixture
def mock_qdrant_wrapper() -> QdrantClientWrapper:
    wrapper = MagicMock(spec=QdrantClientWrapper)
    sample_chunk = RetrievedStatutoryChunk(
        chunk_id="chunk-1",
        act_name="Companies Act, 2013",
        section="Section 185",
        sub_section="(1)",
        title="Loans to Directors",
        content="No company shall advance loans to directors...",
        domain="corporate_law",
        similarity_score=0.88,
        document_id="doc-123",
        document_name="companies_act.pdf",
        page_number=45,
        heading="Chapter XII - Meetings of Board and its Powers",
    )
    wrapper.search_statutes = AsyncMock(return_value=[sample_chunk])
    wrapper.check_health = AsyncMock(return_value=True)
    return wrapper


# ==============================================================================
# 1. Adaptive Top-K & Classification Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_simple_query_routes_to_top_k_simple(mock_nebius_client, phase2_settings):
    mock_nebius_client.create_chat_completion = AsyncMock(
        return_value="CLASSIFICATION: SIMPLE\nDOMAIN: corporate_law\nRATIONALE: Basic definition."
    )
    state: LegalGraphState = {
        "query": "What is the threshold for CSR committee under Section 135?",
        "force_complex": False,
    }
    res = await classify_query_node(state, mock_nebius_client, phase2_settings)
    assert res["query_complexity"] == "simple"
    assert res["selected_model"] == phase2_settings.nemotron_nano_model
    assert res["retrieval_top_k"] == 3
    assert res["retrieval_min_score"] == 0.50


@pytest.mark.asyncio
async def test_complex_query_routes_to_top_k_complex(mock_nebius_client, phase2_settings):
    mock_nebius_client.create_chat_completion = AsyncMock(
        return_value="CLASSIFICATION: COMPLEX\nDOMAIN: corporate_law\nRATIONALE: Interplay between IBC and Companies Act."
    )
    state: LegalGraphState = {
        "query": "Conflict between IBC Section 14 moratorium and Section 185 director loans",
        "force_complex": False,
    }
    res = await classify_query_node(state, mock_nebius_client, phase2_settings)
    assert res["query_complexity"] == "complex"
    assert res["selected_model"] == phase2_settings.nemotron_super_model
    assert res["retrieval_top_k"] == 7
    assert res["retrieval_min_score"] == 0.45


@pytest.mark.asyncio
async def test_configurable_retrieval_parameters(mock_nebius_client):
    custom_settings = Settings(
        nebius_api_key="test-key",
        top_k_simple=4,
        top_k_complex=9,
        min_score_simple=0.55,
        min_score_complex=0.42,
        clerk_dev_mode=True,
    )
    mock_nebius_client.create_chat_completion = AsyncMock(
        return_value="CLASSIFICATION: SIMPLE\nDOMAIN: corporate_law\nRATIONALE: Simple lookup."
    )
    state: LegalGraphState = {"query": "What is OPC?", "force_complex": False}
    simple_res = await classify_query_node(state, mock_nebius_client, custom_settings)
    assert simple_res["retrieval_top_k"] == 4
    assert simple_res["retrieval_min_score"] == 0.55

    state_complex: LegalGraphState = {"query": "Conflict in statutes", "force_complex": True}
    complex_res = await classify_query_node(state_complex, mock_nebius_client, custom_settings)
    assert complex_res["retrieval_top_k"] == 9
    assert complex_res["retrieval_min_score"] == 0.42


# ==============================================================================
# 2. Domain Inference & Preservation Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_domain_inference_when_omitted(mock_nebius_client, phase2_settings):
    mock_nebius_client.create_chat_completion = AsyncMock(
        return_value="CLASSIFICATION: SIMPLE\nDOMAIN: taxation\nRATIONALE: Income tax provision."
    )
    state: LegalGraphState = {
        "query": "What is the penalty for failure to deduct TDS under section 194C?",
        "domain": None,
    }
    res = await classify_query_node(state, mock_nebius_client, phase2_settings)
    assert res["domain_inferred"] is True
    assert res["detected_domain"] == "taxation"


@pytest.mark.asyncio
async def test_explicit_domain_preservation(mock_nebius_client, phase2_settings):
    state: LegalGraphState = {
        "query": "What is the rule under Section 185?",
        "domain": "corporate_law",
    }
    res = await classify_query_node(state, mock_nebius_client, phase2_settings)
    assert res["domain_inferred"] is False
    assert res["detected_domain"] == "corporate_law"


# ==============================================================================
# 3. Compound Filtering & Filter Relaxation Rules
# ==============================================================================


@pytest.mark.asyncio
async def test_explicit_document_id_filtering(mock_nebius_client, mock_qdrant_wrapper, phase2_settings):
    state: LegalGraphState = {
        "query": "Review loan covenants",
        "document_id": "doc-loan-agreement-99",
        "domain": "corporate_law",
        "retrieval_top_k": 3,
        "retrieval_min_score": 0.50,
    }
    res = await retrieve_context_node(state, mock_nebius_client, mock_qdrant_wrapper, phase2_settings)
    assert res["fallback_triggered"] is False
    mock_qdrant_wrapper.search_statutes.assert_called_once_with(
        query_vector=[0.05] * 1024,
        top_k=3,
        min_score=0.50,
        domain_filter="corporate_law",
        document_id_filter="doc-loan-agreement-99",
    )


@pytest.mark.asyncio
async def test_explicit_document_id_with_zero_hits_does_not_relax(mock_nebius_client, mock_qdrant_wrapper, phase2_settings):
    mock_qdrant_wrapper.search_statutes = AsyncMock(return_value=[])
    state: LegalGraphState = {
        "query": "Arbitration clause terms",
        "document_id": "doc-nda-001",
        "domain_inferred": True,  # even if domain was inferred, explicit doc_id MUST NOT relax
        "retrieval_top_k": 3,
        "retrieval_min_score": 0.50,
    }
    res = await retrieve_context_node(state, mock_nebius_client, mock_qdrant_wrapper, phase2_settings)
    assert res["fallback_triggered"] is True
    assert res["filter_relaxed"] is False
    assert mock_qdrant_wrapper.search_statutes.call_count == 1


@pytest.mark.asyncio
async def test_auto_inferred_domain_with_zero_hits_relaxes(mock_nebius_client, mock_qdrant_wrapper, phase2_settings):
    sample_chunk = RetrievedStatutoryChunk(
        chunk_id="chunk-fallback",
        act_name="General Clauses Act, 1897",
        section="Section 3",
        content="Definitions of statutory expressions...",
        similarity_score=0.72,
    )
    # First search with inferred domain returns empty; second relaxed search returns 1 hit
    mock_qdrant_wrapper.search_statutes = AsyncMock(side_effect=[[], [sample_chunk]])

    state: LegalGraphState = {
        "query": "Meaning of month in Indian statutes",
        "detected_domain": "corporate_law",
        "domain_inferred": True,
        "document_id": None,
        "retrieval_top_k": 3,
        "retrieval_min_score": 0.50,
    }
    res = await retrieve_context_node(state, mock_nebius_client, mock_qdrant_wrapper, phase2_settings)
    assert mock_qdrant_wrapper.search_statutes.call_count == 2
    assert res["filter_relaxed"] is True
    assert res["fallback_triggered"] is False
    assert len(res["filtered_chunks"]) == 1


# ==============================================================================
# 4. Score Margin Pruning Tests
# ==============================================================================


def test_score_margin_pruning_drops_distant_candidates():
    chunks = [
        {"chunk_id": "c1", "similarity_score": 0.90, "content": "Top match"},
        {"chunk_id": "c2", "similarity_score": 0.78, "content": "Second match"},
        {"chunk_id": "c3", "similarity_score": 0.55, "content": "Distant match"},  # 0.55 < 0.90 * 0.75 = 0.675
    ]
    pruned = prune_chunks_by_score_margin(chunks, margin_ratio=0.75, min_score=0.40)
    assert len(pruned) == 2
    assert [c["chunk_id"] for c in pruned] == ["c1", "c2"]


def test_score_margin_pruning_always_retains_top_result_if_above_floor():
    chunks = [{"chunk_id": "c1", "similarity_score": 0.52, "content": "Solo match"}]
    pruned = prune_chunks_by_score_margin(chunks, margin_ratio=0.75, min_score=0.50)
    assert len(pruned) == 1
    assert pruned[0]["chunk_id"] == "c1"

    # If top result is below min_score floor, it must be pruned
    pruned_empty = prune_chunks_by_score_margin(chunks, margin_ratio=0.75, min_score=0.60)
    assert len(pruned_empty) == 0


# ==============================================================================
# 5. Evidence Deduplication & Context Formatting Tests
# ==============================================================================


def test_pre_generation_deduplication():
    chunks = [
        {
            "chunk_id": "c1",
            "act_name": "Companies Act, 2013",
            "section": "Section 185",
            "content": "No company shall advance any loan to directors.",
            "similarity_score": 0.82,
            "page_number": 40,
        },
        {
            "chunk_id": "c2",
            "act_name": "Companies Act, 2013",
            "section": "Section 185",
            "content": "Provided that nothing contained in sub-section (1) shall apply to private companies.",
            "similarity_score": 0.89,
            "page_number": 41,
        },
    ]
    deduped = deduplicate_evidence_chunks(chunks)
    assert len(deduped) == 1
    assert deduped[0]["similarity_score"] == 0.89
    assert "No company shall advance" in deduped[0]["content"]
    assert "Provided that nothing" in deduped[0]["content"]


def test_hierarchical_context_grouping_and_exhibits():
    chunks = [
        {
            "chunk_id": "c1",
            "act_name": "Companies Act, 2013",
            "section": "Section 185",
            "title": "Loans to Directors",
            "content": "Prohibition of loans.",
            "similarity_score": 0.88,
        },
        {
            "chunk_id": "c2",
            "act_name": "Income Tax Act, 1961",
            "section": "Section 2(22)(e)",
            "title": "Deemed Dividend",
            "content": "Deemed dividend definition.",
            "similarity_score": 0.81,
        },
    ]
    context_text, included = build_grounded_context_block(chunks, max_chars=5000)
    assert len(included) == 2
    assert "[EXHIBIT 1] Source: Companies Act, 2013 | Provision: Section 185" in context_text
    assert "[EXHIBIT 2] Source: Income Tax Act, 1961 | Provision: Section 2(22)(e)" in context_text


def test_context_character_budget_enforced():
    chunks = [
        {"chunk_id": "c1", "act_name": "Act A", "section": "Sec 1", "content": "A" * 600, "similarity_score": 0.90},
        {"chunk_id": "c2", "act_name": "Act B", "section": "Sec 2", "content": "B" * 600, "similarity_score": 0.85},
    ]
    # Set max_chars to 800: Exhibit 1 fits, Exhibit 2 exceeds and gets cleanly excluded
    context_text, included = build_grounded_context_block(chunks, max_chars=800)
    assert len(included) == 1
    assert "[EXHIBIT 1]" in context_text
    assert "[EXHIBIT 2]" not in context_text


# ==============================================================================
# 6. Evidence-First Safety & Refusal Prompt Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_zero_evidence_refusal_prompt_construction(mock_nebius_client, phase2_settings):
    mock_nebius_client.create_chat_completion = AsyncMock(
        return_value="The indexed corpus does not contain sufficient statutory evidence to answer this question."
    )
    state: LegalGraphState = {
        "query": "What is the penalty for maritime infractions under Act 1900?",
        "selected_model": phase2_settings.nemotron_nano_model,
        "filtered_chunks": [],
        "fallback_triggered": True,
    }
    res = await generate_answer_node(state, mock_nebius_client, phase2_settings)
    assert "The indexed corpus does not contain" in res["raw_answer"]

    # Verify that the instruction explicitly ordered zero fabrication
    call_messages = mock_nebius_client.create_chat_completion.call_args.kwargs["messages"]
    user_prompt = call_messages[1]["content"]
    assert "yielded zero matching statutory provisions" in user_prompt
    assert "Do NOT fabricate section numbers" in user_prompt


@pytest.mark.asyncio
async def test_citations_aligned_with_filtered_evidence_only():
    state: LegalGraphState = {
        "raw_answer": "Final opinion on loan restrictions.",
        # Suppose 3 chunks were originally retrieved, but only 1 survived score margin pruning
        "retrieved_chunks": [
            {"chunk_id": "c1", "act_name": "Act A", "section": "Sec 1", "similarity_score": 0.9, "content": "C1"},
            {"chunk_id": "c2", "act_name": "Act B", "section": "Sec 2", "similarity_score": 0.4, "content": "C2"},
        ],
        "filtered_chunks": [
            {"chunk_id": "c1", "act_name": "Act A", "section": "Sec 1", "similarity_score": 0.9, "content": "C1"},
        ],
    }
    res = await format_citations_node(state)
    assert len(res["citations"]) == 1
    assert res["citations"][0]["act_name"] == "Act A"
    assert res["citations"][0]["section"] == "Sec 1"


def test_citation_metadata_preservation():
    state: LegalGraphState = {
        "raw_answer": "Opinion.",
        "filtered_chunks": [
            {
                "chunk_id": "uuid-1234",
                "act_name": "board_resolution.docx",
                "section": "Clause 4",
                "title": "Director Powers",
                "court_or_authority": None,
                "citation_ref": None,
                "content": "Director powers excerpt.",
                "similarity_score": 0.87654,
                "document_id": "doc-abc-99",
                "document_name": "board_resolution.docx",
                "page_number": 3,
                "heading": "Board Powers",
            }
        ],
    }
    res = format_citations_node(state)
    # format_citations_node is async, await it
    import asyncio
    res_dict = asyncio.run(res)
    assert len(res_dict["citations"]) == 1
    citation = res_dict["citations"][0]
    assert citation["document_id"] == "doc-abc-99"
    assert citation["document_name"] == "board_resolution.docx"
    assert citation["page_number"] == 3
    assert citation["heading"] == "Board Powers"
    assert citation["chunk_id"] == "uuid-1234"
