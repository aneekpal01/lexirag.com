"""LangGraph workflow assembly for the LexiRAG pipeline."""

from typing import Any
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.clients.nebius import NebiusTokenFactoryClient
from app.clients.qdrant import QdrantClientWrapper
from app.core.config import Settings
from app.core.logging import get_logger
from app.rag.nodes import (
    classify_query_node,
    expand_evidence_graph_node,
    format_citations_node,
    generate_answer_node,
    retrieve_context_node,
)
from app.rag.state import LegalGraphState

logger = get_logger(__name__)


def build_legal_rag_graph(
    nebius_client: NebiusTokenFactoryClient,
    qdrant_wrapper: QdrantClientWrapper,
    settings: Settings,
) -> CompiledStateGraph:
    """
    Constructs and compiles the 5-stage LangGraph workflow with dependency-injected clients.
    
    Pipeline Topology:
    [START] 
       │
       ▼
    [classify_query] (Nemotron-3-Nano)
       │
       ▼
    [retrieve_context] (BGE-M3 + Qdrant Cloud)
       │
       ▼
    [expand_evidence_graph] (Controlled Structural Expansion)
       │
       ▼
    [generate_answer] (Routes to Nemotron-3-Nano or Super-120b)
       │
       ▼
    [format_citations] (Indian statutory & precedent formatting)
       │
       ▼
     [END]
    """
    graph_builder = StateGraph(LegalGraphState)

    # Node wrappers injecting clients into the discrete node logic
    async def bound_classify_node(state: LegalGraphState) -> dict[str, Any]:
        return await classify_query_node(state, nebius_client, settings)

    async def bound_retrieve_node(state: LegalGraphState) -> dict[str, Any]:
        return await retrieve_context_node(state, nebius_client, qdrant_wrapper, settings)

    async def bound_expand_node(state: LegalGraphState) -> dict[str, Any]:
        return await expand_evidence_graph_node(state, qdrant_wrapper, settings)

    async def bound_generate_node(state: LegalGraphState) -> dict[str, Any]:
        return await generate_answer_node(state, nebius_client)

    async def bound_format_citations_node(state: LegalGraphState) -> dict[str, Any]:
        return await format_citations_node(state)

    # Register workflow nodes
    graph_builder.add_node("classify_query", bound_classify_node)
    graph_builder.add_node("retrieve_context", bound_retrieve_node)
    graph_builder.add_node("expand_evidence_graph", bound_expand_node)
    graph_builder.add_node("generate_answer", bound_generate_node)
    graph_builder.add_node("format_citations", bound_format_citations_node)

    # Define linear graph edges
    graph_builder.add_edge(START, "classify_query")
    graph_builder.add_edge("classify_query", "retrieve_context")
    graph_builder.add_edge("retrieve_context", "expand_evidence_graph")
    graph_builder.add_edge("expand_evidence_graph", "generate_answer")
    graph_builder.add_edge("generate_answer", "format_citations")
    graph_builder.add_edge("format_citations", END)

    compiled_graph = graph_builder.compile()
    logger.info("LangGraph Indian legal research pipeline successfully compiled")
    return compiled_graph
