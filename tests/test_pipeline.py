"""Unit tests for the LangGraph Indian legal research pipeline nodes and graph."""

from unittest.mock import AsyncMock, MagicMock
import pytest

from app.clients.nebius import NebiusTokenFactoryClient
from app.clients.qdrant import QdrantClientWrapper, RetrievedStatutoryChunk
from app.core.config import Settings
from app.rag.graph import build_legal_rag_graph
from app.rag.nodes import (
    classify_query_node,
    format_citations_node,
    generate_answer_node,
    retrieve_context_node,
)
from app.rag.state import LegalGraphState


@pytest.fixture
def mock_settings() -> Settings:
    return Settings(
        nebius_api_key="test-nebius-key",
        nemotron_nano_model="nvidia/nemotron-3-nano-30b-a3b",
        nemotron_super_model="nvidia/nemotron-3-super-120b-a12b",
        embedding_model="BAAI/bge-m3",
        qdrant_url="http://localhost:6333",
        qdrant_collection_name="indian_legal_corpus",
        top_k_retrieval_limit=3,
        min_similarity_score=0.40,
        clerk_dev_mode=True,
    )


@pytest.fixture
def mock_nebius_client() -> NebiusTokenFactoryClient:
    client = MagicMock(spec=NebiusTokenFactoryClient)
    client.create_chat_completion = AsyncMock(return_value="CLASSIFICATION: SIMPLE\nRATIONALE: Single section inquiry.")
    client.create_embedding = AsyncMock(return_value=[0.05] * 1024)
    return client


@pytest.fixture
def mock_qdrant_wrapper() -> QdrantClientWrapper:
    wrapper = MagicMock(spec=QdrantClientWrapper)
    sample_chunk = RetrievedStatutoryChunk(
        chunk_id="point-1",
        act_name="Companies Act, 2013",
        section="Section 185",
        sub_section="(1)",
        title="Loans to Directors",
        court_or_authority=None,
        citation_ref=None,
        content="No company shall advance any loan to any of its directors...",
        domain="corporate_law",
        similarity_score=0.88,
    )
    wrapper.search_statutes = AsyncMock(return_value=[sample_chunk])
    wrapper.check_health = AsyncMock(return_value=True)
    return wrapper


@pytest.mark.asyncio
async def test_classify_query_node_simple(mock_nebius_client, mock_settings):
    state: LegalGraphState = {
        "query": "What is the penalty under Section 135 for failing to spend CSR funds?",
        "force_complex": False,
    }
    result = await classify_query_node(state, mock_nebius_client, mock_settings)
    assert result["query_complexity"] == "simple"
    assert result["selected_model"] == mock_settings.nemotron_nano_model


@pytest.mark.asyncio
async def test_classify_query_node_complex(mock_nebius_client, mock_settings):
    mock_nebius_client.create_chat_completion = AsyncMock(
        return_value="CLASSIFICATION: COMPLEX\nRATIONALE: Multi-statute conflict between IBC moratorium and PMLA attachment."
    )
    state: LegalGraphState = {
        "query": "How does Section 14 moratorium under IBC reconcile with provisional attachment orders by ED under PMLA?",
        "force_complex": False,
    }
    result = await classify_query_node(state, mock_nebius_client, mock_settings)
    assert result["query_complexity"] == "complex"
    assert result["selected_model"] == mock_settings.nemotron_super_model


@pytest.mark.asyncio
async def test_classify_query_node_forced_complex(mock_nebius_client, mock_settings):
    state: LegalGraphState = {
        "query": "Basic definition of director under Companies Act.",
        "force_complex": True,
    }
    result = await classify_query_node(state, mock_nebius_client, mock_settings)
    assert result["query_complexity"] == "complex"
    assert result["selected_model"] == mock_settings.nemotron_super_model
    # When forced, LLM classification call should be skipped
    mock_nebius_client.create_chat_completion.assert_not_called()


@pytest.mark.asyncio
async def test_retrieve_context_node_success(mock_nebius_client, mock_qdrant_wrapper, mock_settings):
    state: LegalGraphState = {
        "query": "Loans to directors restrictions in private limited companies",
        "domain": "corporate_law",
    }
    result = await retrieve_context_node(state, mock_nebius_client, mock_qdrant_wrapper, mock_settings)
    assert result["fallback_triggered"] is False
    assert len(result["retrieved_chunks"]) == 1
    assert result["retrieved_chunks"][0]["section"] == "Section 185"
    assert len(result["query_embedding"]) == 1024


@pytest.mark.asyncio
async def test_retrieve_context_node_empty_fallback(mock_nebius_client, mock_qdrant_wrapper, mock_settings):
    mock_qdrant_wrapper.search_statutes = AsyncMock(return_value=[])
    state: LegalGraphState = {
        "query": "Exotic non-existent maritime statute inquiry",
    }
    result = await retrieve_context_node(state, mock_nebius_client, mock_qdrant_wrapper, mock_settings)
    assert result["fallback_triggered"] is True
    assert result["retrieved_chunks"] == []


@pytest.mark.asyncio
async def test_generate_answer_node(mock_nebius_client):
    mock_nebius_client.create_chat_completion = AsyncMock(
        return_value="Under Section 185 of the Companies Act 2013, advancing loans is strictly regulated..."
    )
    state: LegalGraphState = {
        "query": "Are loans to directors permissible?",
        "selected_model": "nvidia/nemotron-3-nano-30b-a3b",
        "retrieved_chunks": [
            {
                "act_name": "Companies Act, 2013",
                "section": "Section 185",
                "title": "Loans to Directors",
                "content": "No company shall advance loans...",
                "similarity_score": 0.89,
            }
        ],
        "fallback_triggered": False,
    }
    result = await generate_answer_node(state, mock_nebius_client)
    assert "Under Section 185" in result["raw_answer"]


@pytest.mark.asyncio
async def test_format_citations_node_deduplication():
    state: LegalGraphState = {
        "raw_answer": "Final legal opinion on director liability.",
        "retrieved_chunks": [
            {
                "act_name": "Companies Act, 2013",
                "section": "Section 185",
                "title": "Loans to Directors",
                "content": "First chunk content of Section 185.",
                "similarity_score": 0.90,
            },
            {
                "act_name": "Companies Act, 2013",
                "section": "Section 185",
                "title": "Loans to Directors (sub-clause)",
                "content": "Second chunk duplicate reference to Section 185.",
                "similarity_score": 0.85,
            },
            {
                "act_name": "Income Tax Act, 1961",
                "section": "Section 2(22)(e)",
                "title": "Deemed Dividend",
                "content": "Loans to shareholders considered deemed dividend.",
                "similarity_score": 0.75,
            },
        ],
    }
    result = await format_citations_node(state)
    assert len(result["citations"]) == 2  # De-duplicated Section 185
    assert result["citations"][0]["act_name"] == "Companies Act, 2013"
    assert result["citations"][1]["act_name"] == "Income Tax Act, 1961"
    assert result["final_answer"] == "Final legal opinion on director liability."


@pytest.mark.asyncio
async def test_end_to_end_graph(mock_nebius_client, mock_qdrant_wrapper, mock_settings):
    mock_nebius_client.create_chat_completion = AsyncMock(
        side_effect=[
            "CLASSIFICATION: SIMPLE\nRATIONALE: Single section.",
            "Under Section 185 of Companies Act, 2013...",
        ]
    )

    graph = build_legal_rag_graph(
        nebius_client=mock_nebius_client,
        qdrant_wrapper=mock_qdrant_wrapper,
        settings=mock_settings,
    )

    initial_state: LegalGraphState = {
        "query": "Can a private company grant loan to its managing director?",
        "domain": "corporate_law",
        "jurisdiction": "India",
        "force_complex": False,
    }

    final_state = await graph.ainvoke(initial_state)

    assert final_state["query_complexity"] == "simple"
    assert final_state["selected_model"] == mock_settings.nemotron_nano_model
    assert "Under Section 185" in final_state["final_answer"]
    assert len(final_state["citations"]) == 1
    assert final_state["citations"][0]["section"] == "Section 185"
