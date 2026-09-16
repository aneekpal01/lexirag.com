"""HTTP route controllers for the LexiRAG backend service."""

import time
from typing import Any, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from langgraph.graph.state import CompiledStateGraph

from app.api.dependencies import (
    get_compiled_graph,
    get_ingestion_pipeline,
    get_qdrant_wrapper,
)
from app.auth.clerk import AuthenticatedUser, verify_clerk_token
from app.clients.qdrant import QdrantClientWrapper
from app.core.config import Settings, get_settings
from app.core.constants import ALLOWED_DOCUMENT_EXTENSIONS
from app.core.exceptions import (
    DocumentNotFoundError,
    DocumentPayloadTooLargeError,
    EmptyDocumentError,
    InvalidDocumentError,
)
from app.core.logging import get_logger
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.rag.state import LegalGraphState
from app.schemas.ingestion import (
    DocumentDeleteResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
)
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
        "document_id": request.document_id,
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
        detected_domain=final_state.get("detected_domain") or request.domain,
        filter_relaxed=final_state.get("filter_relaxed", False),
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


# ==============================================================================
# Document Ingestion & Corpus Management Routes (Phase 1)
# ==============================================================================


@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and Index Legal Document",
    description=(
        "Uploads a PDF, DOCX, or TXT document up to 25 MB. Computes canonical SHA-256 hash, "
        "detects legal structural boundaries (Sections, Clauses, Articles), generates BAAI/bge-m3 "
        "dense vectors via Nebius Token Factory, and indexes chunks into Qdrant Cloud."
    ),
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to ingest (.pdf, .docx, .txt)."),
    domain: Optional[str] = Form(default=None, description="Optional legal domain tag."),
    pipeline: DocumentIngestionPipeline = Depends(get_ingestion_pipeline),
    current_user: AuthenticatedUser = Depends(verify_clerk_token),
) -> DocumentUploadResponse:
    """Ingests, parses, chunks, embeds, and indexes a legal document into Qdrant."""
    filename = file.filename or "unknown_document"
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if extension not in ALLOWED_DOCUMENT_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file format '{extension}'. Permitted formats: {', '.join(ALLOWED_DOCUMENT_EXTENSIONS)}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty (0 bytes).",
        )

    receipt = await pipeline.ingest_document(
        file_bytes=file_bytes,
        raw_filename=filename,
        domain=domain,
    )
    return receipt


@router.get(
    "/documents/{document_id}",
    response_model=DocumentStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Indexed Document Status",
    description="Fetches indexing status and chunk breakdowns for a previously ingested document.",
)
async def get_document_status(
    document_id: str,
    qdrant: QdrantClientWrapper = Depends(get_qdrant_wrapper),
    current_user: AuthenticatedUser = Depends(verify_clerk_token),
) -> DocumentStatusResponse:
    """Checks whether a document is present in Qdrant and previews its indexed chunks."""
    chunks = await qdrant.get_document_chunks(document_id=document_id, limit=20)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' was not found in the indexed corpus.",
        )

    sample_chunks = [
        {
            "chunk_id": c.get("chunk_id"),
            "chunk_index": c.get("chunk_index"),
            "section": c.get("section"),
            "page_number": c.get("page_number"),
            "heading": c.get("heading"),
            "token_count": c.get("token_count", 0),
        }
        for c in chunks
    ]

    filename = chunks[0].get("document_name") if chunks else None
    return DocumentStatusResponse(
        document_id=document_id,
        filename=filename,
        total_chunks=len(chunks),
        status="indexed",
        sample_chunks=sample_chunks,
    )


@router.delete(
    "/documents/{document_id}",
    response_model=DocumentDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Purge Document From Corpus",
    description="Deletes all indexed vector chunks associated with a document_id from Qdrant Cloud.",
)
async def delete_document(
    document_id: str,
    qdrant: QdrantClientWrapper = Depends(get_qdrant_wrapper),
    current_user: AuthenticatedUser = Depends(verify_clerk_token),
) -> DocumentDeleteResponse:
    """Purges all vector records belonging to the target document_id."""
    # Check if document exists first
    existing = await qdrant.get_document_chunks(document_id=document_id, limit=1)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cannot delete document '{document_id}': Document does not exist in corpus.",
        )

    await qdrant.delete_document(document_id=document_id)
    return DocumentDeleteResponse(
        document_id=document_id,
        deleted=True,
        message=f"Document '{document_id}' and all associated vectors were successfully purged.",
    )
