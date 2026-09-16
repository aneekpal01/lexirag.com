"""Qdrant Cloud client wrapper for Indian statutory and case law vector retrieval."""

from typing import Any, Optional
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.core.config import Settings
from app.core.constants import (
    DEFAULT_MIN_SIMILARITY_SCORE,
    DEFAULT_TOP_K_RETRIEVAL,
    QDRANT_REQUEST_TIMEOUT_SECONDS,
)
from app.core.exceptions import (
    MalformedQdrantPayloadError,
    QdrantServiceError,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


class RetrievedStatutoryChunk(BaseModel):
    """Normalized representation of a statutory section or judicial excerpt."""

    chunk_id: str
    act_name: str
    section: str
    sub_section: Optional[str] = None
    title: Optional[str] = None
    court_or_authority: Optional[str] = None
    citation_ref: Optional[str] = None
    content: str
    domain: Optional[str] = None
    similarity_score: float


class QdrantClientWrapper:
    """Manages asynchronous communication and vector similarity queries with Qdrant Cloud."""

    def __init__(self, settings: Settings, client: Optional[AsyncQdrantClient] = None):
        self.settings = settings
        self.client = client or AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=QDRANT_REQUEST_TIMEOUT_SECONDS,
        )

    def _parse_payload(self, point_id: Any, payload: Optional[dict[str, Any]], score: float) -> RetrievedStatutoryChunk:
        """
        Extracts statutory fields defensively to prevent failures on malformed vectors.
        """
        if not payload:
            raise MalformedQdrantPayloadError(
                f"Qdrant point {point_id} returned empty payload dictionary"
            )

        content = payload.get("content") or payload.get("text") or payload.get("chunk_text")
        if not content or not str(content).strip():
            raise MalformedQdrantPayloadError(
                f"Qdrant point {point_id} is missing mandatory statutory content field"
            )

        act_name = str(payload.get("act_name") or payload.get("statute") or "Statute Reference Unavailable")
        section = str(payload.get("section") or payload.get("provision") or "Provision Unspecified")

        return RetrievedStatutoryChunk(
            chunk_id=str(point_id),
            act_name=act_name,
            section=section,
            sub_section=payload.get("sub_section"),
            title=payload.get("title") or payload.get("heading"),
            court_or_authority=payload.get("court_or_authority") or payload.get("court"),
            citation_ref=payload.get("citation_ref") or payload.get("citation"),
            content=str(content).strip(),
            domain=payload.get("domain"),
            similarity_score=float(score),
        )

    async def search_statutes(
        self,
        query_vector: list[float],
        top_k: int = DEFAULT_TOP_K_RETRIEVAL,
        min_score: float = DEFAULT_MIN_SIMILARITY_SCORE,
        domain_filter: Optional[str] = None,
    ) -> list[RetrievedStatutoryChunk]:
        """
        Executes dense vector similarity search against the pre-embedded legal corpus.
        """
        logger.debug(
            "Executing Qdrant vector search | collection: %s | top_k: %d | min_score: %.2f",
            self.settings.qdrant_collection_name,
            top_k,
            min_score,
        )

        try:
            # We use search or query_points depending on qdrant-client version
            if hasattr(self.client, "search"):
                scored_points = await self.client.search(
                    collection_name=self.settings.qdrant_collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    score_threshold=min_score,
                )
            else:
                query_result = await self.client.query_points(
                    collection_name=self.settings.qdrant_collection_name,
                    query=query_vector,
                    limit=top_k,
                    score_threshold=min_score,
                )
                scored_points = query_result.points

        except (ResponseHandlingException, UnexpectedResponse) as exc:
            logger.error("Qdrant cloud responded with an API error: %s", exc)
            raise QdrantServiceError(f"Qdrant vector query failed: {exc}") from exc
        except Exception as exc:
            logger.exception("Unexpected failure when querying Qdrant cluster")
            raise QdrantServiceError(f"Failed to communicate with Qdrant: {exc}") from exc

        parsed_chunks: list[RetrievedStatutoryChunk] = []
        for point in scored_points:
            try:
                parsed_chunk = self._parse_payload(
                    point_id=point.id,
                    payload=point.payload,
                    score=point.score,
                )
                parsed_chunks.append(parsed_chunk)
            except MalformedQdrantPayloadError as payload_err:
                logger.warning(
                    "Skipping malformed vector point %s: %s",
                    point.id,
                    payload_err.message,
                )
                continue

        logger.info(
            "Retrieved %d valid statutory chunks from Qdrant (surpassed score floor %.2f)",
            len(parsed_chunks),
            min_score,
        )
        return parsed_chunks

    async def check_health(self) -> bool:
        """Verifies connectivity to Qdrant cluster and existence of target collection."""
        try:
            collections_response = await self.client.get_collections()
            existing_collections = [c.name for c in collections_response.collections]
            return self.settings.qdrant_collection_name in existing_collections
        except Exception as exc:
            logger.warning("Qdrant health check probe failed: %s", exc)
            return False
