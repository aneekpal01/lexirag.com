"""HTTP route controllers for the LexiRAG backend service."""

import time
from typing import Any
from fastapi import APIRouter, Depends, status
from langgraph.graph.state import CompiledStateGraph

from app.api.dependencies import get_compiled_graph, get_qdrant_wrapper
from app.auth.clerk import AuthenticatedUser, verify_clerk_token
from app.clients.qdrant import QdrantClientWrapper
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.rag.state import LegalGraphState
from app.schemas.query import LegalCitation, LegalQueryRequest, LegalQueryResponse

logger = get_logger(__name__)
router = APIRouter(tags=["Legal Research RAG"])


@router.post(
    "/query",
    response_model=LegalQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute Indian Legal Research Query",
    description=(
        "Processes an Indian corporate, tax, or regulatory legal question through the 4-stage "
        "LangGraph pipeline. Routes between Nemotron Nano and Nemotron Super reasoning models on "
        "Nebius Token Factory based on statutory complexity."
    ),
)
async def query_legal_corpus(
    request: LegalQueryRequest,
    graph: CompiledStateGraph = Depends(get_compiled_graph),
    current_user: AuthenticatedUser = Depends(verify_clerk_token),
) -> LegalQueryResponse:
    """Executes the legal RAG workflow and returns a grounded opinion with statutory citations."""
    start_time = time.perf_counter()
    logger.info(
        "Initiating legal research pipeline | user_id: %s | query_len: %d chars",
        current_user.user_id,
        len(request.query),
    )

    initial_state: LegalGraphState = {
        "query": request.query,
        "domain": request.domain,
        "jurisdiction": request.jurisdiction,
        "force_complex": request.force_complex_reasoning,
    }

    # Execute the compiled LangGraph workflow
    final_state: LegalGraphState = await graph.ainvoke(initial_state)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Parse and validate citations into response schema
    raw_citations = final_state.get("citations", [])
    structured_citations = [LegalCitation(**citation) for citation in raw_citations]

    response_payload = LegalQueryResponse(
        answer=final_state.get("final_answer", ""),
        citations=structured_citations,
        complexity=final_state.get("query_complexity", "simple"),
        model_used=final_state.get("selected_model", "unknown"),
        retrieved_chunks_count=len(final_state.get("retrieved_chunks", [])),
        execution_time_ms=round(elapsed_ms, 2),
        fallback_triggered=final_state.get("fallback_triggered", False),
    )

    logger.info(
        "Completed legal research pipeline | user: %s | model: %s | citations: %d | elapsed: %.2fms",
        current_user.user_id,
        response_payload.model_used,
        len(response_payload.citations),
        elapsed_ms,
    )

    return response_payload


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="System Health & Readiness Probe",
    description="Validates runtime configuration and connectivity to external vector services.",
)
async def health_check(
    settings: Settings = Depends(get_settings),
    qdrant: QdrantClientWrapper = Depends(get_qdrant_wrapper),
) -> dict[str, Any]:
    """Health check endpoint for Kubernetes liveness/readiness probes."""
    qdrant_healthy = await qdrant.check_health()

    return {
        "status": "healthy" if qdrant_healthy else "degraded",
        "service": "LexiRAG SaaS Backend",
        "environment": settings.environment,
        "nebius_token_factory": {
            "base_url": settings.nebius_base_url,
            "models": {
                "nano_tier": settings.nemotron_nano_model,
                "super_tier": settings.nemotron_super_model,
                "embedding": settings.embedding_model,
            },
        },
        "qdrant": {
            "collection": settings.qdrant_collection_name,
            "reachable": qdrant_healthy,
        },
        "auth": {
            "clerk_dev_mode": settings.clerk_dev_mode,
        },
    }
