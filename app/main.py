"""FastAPI application entrypoint and exception handling configuration."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router as legal_router
from app.core.config import get_settings
from app.core.exceptions import (
    AuthenticationError,
    InvalidLegalQueryError,
    NebiusExtractionError,
    NebiusRateLimitError,
    NebiusServiceError,
    NebiusTimeoutError,
    QdrantServiceError,
)
from app.core.logging import get_logger, setup_application_logging
from app.schemas.errors import ErrorResponse

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager for startup and shutdown routines."""
    settings = get_settings()
    setup_application_logging(log_level=settings.log_level)
    logger.info("Initializing LexiRAG SaaS Backend | environment: %s", settings.environment)
    logger.info("Targeting Nebius Token Factory: %s", settings.nebius_base_url)
    logger.info("Nemotron Models: [Nano: %s, Super: %s]", settings.nemotron_nano_model, settings.nemotron_super_model)
    logger.info("Qdrant Cluster: %s | Collection: %s", settings.qdrant_url, settings.qdrant_collection_name)
    yield
    logger.info("Shutting down LexiRAG backend services gracefully")


def create_application() -> FastAPI:
    """Creates and configures the production FastAPI instance."""
    app = FastAPI(
        title="LexiRAG - Indian Legal Research RAG Backend",
        description=(
            "Production RAG backend for Indian corporate, tax, labor, and regulatory jurisprudence. "
            "Orchestrates retrieval from Qdrant Cloud (BGE-M3) and reasoning via NVIDIA Nemotron "
            "models on Nebius Token Factory."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS configuration for SaaS frontend (e.g. Next.js dashboard)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routes
    app.include_router(legal_router, prefix="/api/v1")
    # Also mount at root level for convenience (POST /query)
    app.include_router(legal_router)

    # --------------------------------------------------------------------------
    # Global Domain Exception Handlers
    # --------------------------------------------------------------------------

    @app.exception_handler(NebiusRateLimitError)
    async def handle_nebius_rate_limit(request: Request, exc: NebiusRateLimitError) -> JSONResponse:
        logger.error("Nebius rate limit encountered on path %s: %s", request.url.path, exc.message)
        headers = {}
        if exc.retry_after is not None:
            headers["Retry-After"] = str(exc.retry_after)

        payload = ErrorResponse(
            error_code="NEBIUS_RATE_LIMITED",
            message=exc.message,
            details=exc.details,
        )
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=payload.model_dump(),
            headers=headers,
        )

    @app.exception_handler(NebiusTimeoutError)
    async def handle_nebius_timeout(request: Request, exc: NebiusTimeoutError) -> JSONResponse:
        logger.error("Nebius timeout encountered on path %s: %s", request.url.path, exc.message)
        payload = ErrorResponse(
            error_code="NEBIUS_TIMEOUT",
            message=exc.message,
            details=exc.details,
        )
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content=payload.model_dump(),
        )

    @app.exception_handler(NebiusExtractionError)
    async def handle_nebius_extraction_error(request: Request, exc: NebiusExtractionError) -> JSONResponse:
        logger.error("Nebius reasoning extraction failed: %s", exc.message)
        payload = ErrorResponse(
            error_code="NEBIUS_EXTRACTION_FAILURE",
            message=exc.message,
            details={"hint": "Verify that Nemotron output was returned in reasoning_content or content."},
        )
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=payload.model_dump(),
        )

    @app.exception_handler(NebiusServiceError)
    async def handle_nebius_service_error(request: Request, exc: NebiusServiceError) -> JSONResponse:
        logger.error("Nebius service error on path %s: %s", request.url.path, exc.message)
        payload = ErrorResponse(
            error_code="NEBIUS_SERVICE_FAILURE",
            message=exc.message,
            details=exc.details,
        )
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=payload.model_dump(),
        )

    @app.exception_handler(QdrantServiceError)
    async def handle_qdrant_service_error(request: Request, exc: QdrantServiceError) -> JSONResponse:
        logger.error("Qdrant service error on path %s: %s", request.url.path, exc.message)
        payload = ErrorResponse(
            error_code="QDRANT_UNAVAILABLE",
            message=exc.message,
            details=exc.details,
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=payload.model_dump(),
        )

    @app.exception_handler(AuthenticationError)
    async def handle_auth_error(request: Request, exc: AuthenticationError) -> JSONResponse:
        logger.warning("Unauthorized access attempt on %s: %s", request.url.path, exc.message)
        payload = ErrorResponse(
            error_code="AUTHENTICATION_FAILED",
            message=exc.message,
            details=exc.details,
        )
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content=payload.model_dump(),
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.exception_handler(InvalidLegalQueryError)
    async def handle_invalid_query_error(request: Request, exc: InvalidLegalQueryError) -> JSONResponse:
        logger.warning("Invalid query rejected on %s: %s", request.url.path, exc.message)
        payload = ErrorResponse(
            error_code="INVALID_QUERY",
            message=exc.message,
            details=exc.details,
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=payload.model_dump(),
        )

    return app


app = create_application()
