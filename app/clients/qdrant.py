"""Qdrant Cloud client wrapper for Indian statutory and case law vector retrieval."""

from typing import Any, Optional
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models
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
    """Normalized representation of a statutory section, contract clause, or document excerpt."""

    chunk_id: str
    act_name: str
    section: str
    sub_section: Optional[str] = None
    title: Optional[str] = None
    court_or_authority: Optional[str] = None
    citation_ref: Optional[str] = None
    content: str
    domain: Optional[str] = None
    similarity_score: Optional[float] = None

    # Extended Phase 1 Document Metadata
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    source: Optional[str] = None
    page_number: Optional[int] = None
    heading: Optional[str] = None
    file_type: Optional[str] = None
    chunk_index: Optional[int] = None

    # Phase 3 Adjacency Graph Pointers
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None


class QdrantClientWrapper:
    """Manages asynchronous communication and vector similarity queries with Qdrant Cloud."""

    def __init__(self, settings: Settings, client: Optional[AsyncQdrantClient] = None):
        self.settings = settings
        self.client = client or AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=QDRANT_REQUEST_TIMEOUT_SECONDS,
        )

    def _parse_payload(
        self,
        point_id: Any,
        payload: Optional[dict[str, Any]],
        score: Optional[float] = None,
    ) -> RetrievedStatutoryChunk:
        """
        Extracts statutory fields defensively to prevent failures on malformed vectors.
        Supports both statutory chunks and newly ingested general legal documents.
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

        act_name = str(
            payload.get("act_name")
            or payload.get("document_name")
            or payload.get("statute")
            or "Statute Reference Unavailable"
        )
        section = str(
            payload.get("section")
            or payload.get("heading")
            or payload.get("provision")
            or "General Provision"
        )

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
            similarity_score=float(score) if score is not None else None,
            document_id=payload.get("document_id"),
            document_name=payload.get("document_name"),
            source=payload.get("source"),
            page_number=payload.get("page_number"),
            heading=payload.get("heading"),
            file_type=payload.get("file_type"),
            chunk_index=payload.get("chunk_index"),
            prev_chunk_id=payload.get("prev_chunk_id"),
            next_chunk_id=payload.get("next_chunk_id"),
        )

    async def search_statutes(
        self,
        query_vector: list[float],
        top_k: int = DEFAULT_TOP_K_RETRIEVAL,
        min_score: float = DEFAULT_MIN_SIMILARITY_SCORE,
        domain_filter: Optional[str] = None,
        document_id_filter: Optional[str] = None,
    ) -> list[RetrievedStatutoryChunk]:
        """
        Executes dense vector similarity search against the pre-embedded legal corpus.
        Supports compound filtering on domain and document_id.
        """
        logger.debug(
            "Executing Qdrant vector search | collection: %s | top_k: %d | min_score: %.2f | domain: %s | doc_id: %s",
            self.settings.qdrant_collection_name,
            top_k,
            min_score,
            domain_filter,
            document_id_filter,
        )

        filter_conditions: list[models.FieldCondition] = []
        if domain_filter and domain_filter.strip():
            filter_conditions.append(
                models.FieldCondition(
                    key="domain",
                    match=models.MatchValue(value=domain_filter.strip()),
                )
            )
        if document_id_filter and document_id_filter.strip():
            filter_conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id_filter.strip()),
                )
            )

        query_filter: Optional[models.Filter] = (
            models.Filter(must=filter_conditions) if filter_conditions else None
        )

        try:
            # We use search or query_points depending on qdrant-client version
            if hasattr(self.client, "search"):
                scored_points = await self.client.search(
                    collection_name=self.settings.qdrant_collection_name,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                    score_threshold=min_score,
                )
            else:
                query_result = await self.client.query_points(
                    collection_name=self.settings.qdrant_collection_name,
                    query=query_vector,
                    query_filter=query_filter,
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

    async def ensure_collection_exists(
        self,
        collection_name: Optional[str] = None,
        vector_size: int = 1024,
    ) -> bool:
        """Provisions target collection with Cosine distance if it does not already exist."""
        target_name = collection_name or self.settings.qdrant_collection_name
        try:
            collections_res = await self.client.get_collections()
            existing = [c.name for c in collections_res.collections]
            if target_name not in existing:
                logger.info("Creating Qdrant collection '%s' with vector size %d", target_name, vector_size)
                await self.client.create_collection(
                    collection_name=target_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
            return True
        except Exception as exc:
            logger.error("Failed to ensure Qdrant collection exists: %s", exc)
            raise QdrantServiceError(f"Failed to provision collection '{target_name}': {exc}") from exc

    async def upsert_points(
        self,
        points: list[models.PointStruct],
        collection_name: Optional[str] = None,
    ) -> None:
        """Batch upserts vector points into target Qdrant collection."""
        if not points:
            return
        target_name = collection_name or self.settings.qdrant_collection_name
        try:
            await self.client.upsert(
                collection_name=target_name,
                points=points,
            )
            logger.info("Upserted %d points to Qdrant collection '%s'", len(points), target_name)
        except Exception as exc:
            logger.error("Failed to upsert points to Qdrant collection '%s': %s", target_name, exc)
            raise QdrantServiceError(f"Point upsert failed: {exc}") from exc

    async def delete_document(
        self,
        document_id: str,
        collection_name: Optional[str] = None,
    ) -> bool:
        """Deletes all chunks associated with a document_id."""
        target_name = collection_name or self.settings.qdrant_collection_name
        try:
            doc_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )
            await self.client.delete(
                collection_name=target_name,
                points_selector=doc_filter,
            )
            logger.info("Deleted document vectors for document_id '%s' from '%s'", document_id, target_name)
            return True
        except Exception as exc:
            logger.error("Failed to delete document '%s' from Qdrant: %s", document_id, exc)
            raise QdrantServiceError(f"Failed to delete document {document_id}: {exc}") from exc

    async def get_document_chunks(
        self,
        document_id: str,
        limit: int = 100,
        collection_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Retrieves point payloads belonging to a specific document_id."""
        target_name = collection_name or self.settings.qdrant_collection_name
        try:
            doc_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )
            res = await self.client.scroll(
                collection_name=target_name,
                scroll_filter=doc_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            points, _ = res
            return [p.payload for p in points if p.payload]
        except Exception as exc:
            logger.warning("Failed to scroll document points for '%s': %s", document_id, exc)
            return []

    async def get_chunks_by_ids(
        self,
        chunk_ids: list[str],
        collection_name: Optional[str] = None,
    ) -> list[RetrievedStatutoryChunk]:
        """
        Directly retrieves specific vector points by deterministic point IDs without vector search.
        Qdrant direct point-ID retrieval avoids vector similarity search for known neighboring chunk IDs.
        """
        if not chunk_ids:
            return []
        target_name = collection_name or self.settings.qdrant_collection_name
        try:
            points = await self.client.retrieve(
                collection_name=target_name,
                ids=chunk_ids,
                with_payload=True,
                with_vectors=False,
            )
            chunks: list[RetrievedStatutoryChunk] = []
            for point in points:
                try:
                    chunk = self._parse_payload(
                        point_id=point.id,
                        payload=point.payload,
                        score=None,
                    )
                    chunks.append(chunk)
                except MalformedQdrantPayloadError as err:
                    logger.warning("Skipping malformed neighbor chunk point %s: %s", point.id, err.message)
                    continue
            logger.debug("Retrieved %d / %d chunks by point IDs from collection '%s'", len(chunks), len(chunk_ids), target_name)
            return chunks
        except Exception as exc:
            logger.error("Failed to retrieve points by ID from Qdrant: %s", exc)
            raise QdrantServiceError(f"Point lookup by ID failed: {exc}") from exc

    async def check_health(self) -> bool:
        """Verifies connectivity to Qdrant cluster and existence of target collection."""
        try:
            collections_response = await self.client.get_collections()
            existing_collections = [c.name for c in collections_response.collections]
            return self.settings.qdrant_collection_name in existing_collections
        except Exception as exc:
            logger.warning("Qdrant health check probe failed: %s", exc)
            return False
