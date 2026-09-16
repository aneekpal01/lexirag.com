"""FastAPI dependency injection providers."""

from functools import lru_cache
from fastapi import Depends
from langgraph.graph.state import CompiledStateGraph

from app.clients.nebius import NebiusTokenFactoryClient
from app.clients.qdrant import QdrantClientWrapper
from app.core.config import Settings, get_settings
from app.rag.graph import build_legal_rag_graph


def get_nebius_client(
    settings: Settings = Depends(get_settings),
) -> NebiusTokenFactoryClient:
    """Provides an active Nebius Token Factory client."""
    return NebiusTokenFactoryClient(settings=settings)


def get_qdrant_wrapper(
    settings: Settings = Depends(get_settings),
) -> QdrantClientWrapper:
    """Provides a configured Qdrant Cloud client wrapper."""
    return QdrantClientWrapper(settings=settings)


def get_compiled_graph(
    nebius_client: NebiusTokenFactoryClient = Depends(get_nebius_client),
    qdrant_wrapper: QdrantClientWrapper = Depends(get_qdrant_wrapper),
    settings: Settings = Depends(get_settings),
) -> CompiledStateGraph:
    """Builds and provides a compiled LangGraph pipeline instance."""
    return build_legal_rag_graph(
        nebius_client=nebius_client,
        qdrant_wrapper=qdrant_wrapper,
        settings=settings,
    )


def get_ingestion_pipeline(
    nebius_client: NebiusTokenFactoryClient = Depends(get_nebius_client),
    qdrant_wrapper: QdrantClientWrapper = Depends(get_qdrant_wrapper),
    settings: Settings = Depends(get_settings),
):
    """Provides a configured DocumentIngestionPipeline instance."""
    from app.ingestion.pipeline import DocumentIngestionPipeline
    return DocumentIngestionPipeline(
        nebius_client=nebius_client,
        qdrant_wrapper=qdrant_wrapper,
        settings=settings,
    )

