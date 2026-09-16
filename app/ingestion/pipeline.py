"""Document ingestion pipeline orchestrating parsing, chunking, embedding, and indexing."""

import os
import re
from typing import Optional
from qdrant_client import models

from app.clients.nebius import NebiusTokenFactoryClient
from app.clients.qdrant import QdrantClientWrapper
from app.core.config import Settings
from app.core.constants import QDRANT_VECTOR_DIMENSION
from app.core.exceptions import (
    DocumentPayloadTooLargeError,
    InvalidDocumentError,
    QdrantServiceError,
)
from app.core.logging import get_logger
from app.ingestion.chunker import StructureAwareLegalChunker
from app.ingestion.parsers import compute_sha256, parse_document
from app.schemas.ingestion import DocumentUploadResponse

logger = get_logger(__name__)


def sanitize_filename(filename: str) -> str:
    """Removes path traversals and illegal characters from uploaded filenames."""
    base_name = os.path.basename(filename).strip()
    clean_name = re.sub(r"[^a-zA-Z0-9_\.\- ]", "_", base_name)
    return clean_name or "uploaded_document"


class DocumentIngestionPipeline:
    """Coordinates the evidence-first ingestion process from raw bytes to Qdrant vectors."""

    def __init__(
        self,
        nebius_client: NebiusTokenFactoryClient,
        qdrant_wrapper: QdrantClientWrapper,
        settings: Settings,
    ):
        self.nebius_client = nebius_client
        self.qdrant_wrapper = qdrant_wrapper
        self.settings = settings
        self.chunker = StructureAwareLegalChunker(
            chunk_size_chars=settings.chunk_size_chars,
            chunk_overlap_chars=settings.chunk_overlap_chars,
        )

    async def ingest_document(
        self,
        file_bytes: bytes,
        raw_filename: str,
        domain: Optional[str] = None,
    ) -> DocumentUploadResponse:
        """
        Executes end-to-end ingestion pipeline:
        1. File size enforcement
        2. Filename sanitization
        3. SHA-256 duplicate verification
        4. Page/structure parsing
        5. Legal boundary chunking
        6. Nebius Token Factory BGE-M3 batch embedding
        7. Qdrant Cloud point upsert with rollback safety
        """
        # 1. Enforce payload size limit
        file_size = len(file_bytes)
        if file_size > self.settings.max_upload_size_bytes:
            raise DocumentPayloadTooLargeError(
                f"File size ({file_size / (1024 * 1024):.2f} MB) exceeds limit of "
                f"{self.settings.max_upload_size_bytes / (1024 * 1024):.0f} MB."
            )

        # 2. Sanitize filename
        filename = sanitize_filename(raw_filename)
        doc_id = compute_sha256(file_bytes)

        logger.info("Starting document ingestion | doc_id: %s | filename: %s | size: %d bytes", doc_id, filename, file_size)

        # 3. Duplicate detection: check if points for this document already exist
        existing_chunks = await self.qdrant_wrapper.get_document_chunks(doc_id, limit=5)
        if existing_chunks:
            logger.info("Document '%s' (ID: %s) already indexed. Skipping redundant embedding.", filename, doc_id)
            detected_sections = list({c.get("section") for c in existing_chunks if c.get("section")})
            return DocumentUploadResponse(
                document_id=doc_id,
                filename=filename,
                file_type=filename.rsplit(".", 1)[-1].lower(),
                file_size_bytes=file_size,
                total_chunks=len(existing_chunks),
                detected_sections=detected_sections,
                status="already_indexed",
                message="Document has already been ingested. Vector generation was skipped to prevent duplicate records.",
            )

        # 4. Parse document structure
        parsed_doc = parse_document(file_bytes=file_bytes, filename=filename, domain=domain)

        # 5. Structure-aware chunking
        chunks = self.chunker.chunk_document(parsed_doc)
        if not chunks:
            raise InvalidDocumentError(f"Document '{filename}' produced zero valid chunks.")

        detected_sections = list({chunk.section for chunk in chunks if chunk.section != "General Provision"})

        # 6. Ensure Qdrant collection exists
        await self.qdrant_wrapper.ensure_collection_exists(vector_size=QDRANT_VECTOR_DIMENSION)

        # 7. Generate dense vectors in batches via Nebius Token Factory
        chunk_texts = [chunk.text for chunk in chunks]
        try:
            vectors = await self.nebius_client.create_embeddings_batch(
                texts=chunk_texts,
                batch_size=self.settings.embedding_batch_size,
            )
        except Exception as exc:
            logger.error("Nebius batch embedding failed during ingestion of '%s': %s", filename, exc)
            raise

        # 8. Assemble Qdrant points
        points: list[models.PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            points.append(
                models.PointStruct(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload=chunk.to_qdrant_payload(),
                )
            )

        # 9. Atomic upsert with rollback safety
        try:
            await self.qdrant_wrapper.upsert_points(points=points)
        except Exception as exc:
            logger.error("Qdrant upsert failed. Rolling back partial vectors for doc_id '%s': %s", doc_id, exc)
            try:
                await self.qdrant_wrapper.delete_document(doc_id)
            except Exception as rollback_exc:
                logger.error("Rollback cleanup failed for doc_id '%s': %s", doc_id, rollback_exc)
            raise QdrantServiceError(f"Failed to upsert document vectors into Qdrant: {exc}") from exc

        logger.info(
            "Document successfully indexed | doc_id: %s | filename: %s | total_chunks: %d",
            doc_id,
            filename,
            len(chunks),
        )

        return DocumentUploadResponse(
            document_id=doc_id,
            filename=filename,
            file_type=parsed_doc.file_type,
            file_size_bytes=file_size,
            total_chunks=len(chunks),
            detected_sections=detected_sections,
            status="indexed",
            message="Document successfully parsed, chunked, and indexed into Qdrant.",
        )
