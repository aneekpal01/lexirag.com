"""Unit and integration tests for DocumentIngestionPipeline."""

from unittest.mock import AsyncMock, MagicMock
import pytest

from app.clients.nebius import NebiusTokenFactoryClient
from app.clients.qdrant import QdrantClientWrapper
from app.core.config import Settings
from app.core.exceptions import DocumentPayloadTooLargeError, QdrantServiceError
from app.ingestion.pipeline import DocumentIngestionPipeline, sanitize_filename
from tests.test_ingestion_parsers import SAMPLE_PDF_BYTES


@pytest.fixture
def mock_settings() -> Settings:
    return Settings(
        max_upload_size_bytes=10000,  # 10 KB for test
        chunk_size_chars=400,
        chunk_overlap_chars=50,
        embedding_batch_size=4,
        qdrant_collection_name="test_legal_collection",
    )


@pytest.fixture
def mock_clients():
    mock_nebius = MagicMock(spec=NebiusTokenFactoryClient)
    # Return 1024-dim dummy vectors
    mock_nebius.create_embeddings_batch = AsyncMock(
        side_effect=lambda texts, batch_size=None: [[0.02] * 1024 for _ in texts]
    )

    mock_qdrant = MagicMock(spec=QdrantClientWrapper)
    mock_qdrant.ensure_collection_exists = AsyncMock(return_value=True)
    mock_qdrant.upsert_points = AsyncMock(return_value=None)
    mock_qdrant.get_document_chunks = AsyncMock(return_value=[])
    mock_qdrant.delete_document = AsyncMock(return_value=True)

    return mock_nebius, mock_qdrant


def test_sanitize_filename():
    assert sanitize_filename("../../../etc/passwd") == "passwd"
    assert sanitize_filename("Companies Act (2013)#Section 185!.pdf") == "Companies Act _2013__Section 185_.pdf"
    assert sanitize_filename("normal_doc.pdf") == "normal_doc.pdf"


@pytest.mark.asyncio
async def test_pipeline_ingests_document_successfully(mock_settings, mock_clients):
    mock_nebius, mock_qdrant = mock_clients
    pipeline = DocumentIngestionPipeline(
        nebius_client=mock_nebius,
        qdrant_wrapper=mock_qdrant,
        settings=mock_settings,
    )

    receipt = await pipeline.ingest_document(
        file_bytes=SAMPLE_PDF_BYTES,
        raw_filename="statute.pdf",
        domain="corporate_law",
    )

    assert receipt.status == "indexed"
    assert receipt.filename == "statute.pdf"
    assert receipt.total_chunks >= 1
    assert len(receipt.document_id) == 64
    mock_qdrant.ensure_collection_exists.assert_called_once()
    mock_nebius.create_embeddings_batch.assert_called_once()
    mock_qdrant.upsert_points.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_skips_duplicate_document(mock_settings, mock_clients):
    mock_nebius, mock_qdrant = mock_clients
    # Simulate that document was already indexed
    mock_qdrant.get_document_chunks = AsyncMock(return_value=[{"chunk_id": "c-1", "section": "Section 185"}])

    pipeline = DocumentIngestionPipeline(
        nebius_client=mock_nebius,
        qdrant_wrapper=mock_qdrant,
        settings=mock_settings,
    )

    receipt = await pipeline.ingest_document(
        file_bytes=SAMPLE_PDF_BYTES,
        raw_filename="duplicate.pdf",
    )

    assert receipt.status == "already_indexed"
    assert "already been ingested" in receipt.message
    # Embeddings and upserts should NOT have been called
    mock_nebius.create_embeddings_batch.assert_not_called()
    mock_qdrant.upsert_points.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_rollback_on_upsert_failure(mock_settings, mock_clients):
    mock_nebius, mock_qdrant = mock_clients
    mock_qdrant.upsert_points = AsyncMock(side_effect=Exception("Qdrant write timeout"))

    pipeline = DocumentIngestionPipeline(
        nebius_client=mock_nebius,
        qdrant_wrapper=mock_qdrant,
        settings=mock_settings,
    )

    with pytest.raises(QdrantServiceError):
        await pipeline.ingest_document(
            file_bytes=SAMPLE_PDF_BYTES,
            raw_filename="fail.pdf",
        )

    # Rollback delete_document must have been called
    mock_qdrant.delete_document.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_rejects_payload_exceeding_size_limit(mock_settings, mock_clients):
    mock_nebius, mock_qdrant = mock_clients
    pipeline = DocumentIngestionPipeline(
        nebius_client=mock_nebius,
        qdrant_wrapper=mock_qdrant,
        settings=mock_settings,
    )

    huge_bytes = b"0" * (mock_settings.max_upload_size_bytes + 100)

    with pytest.raises(DocumentPayloadTooLargeError):
        await pipeline.ingest_document(
            file_bytes=huge_bytes,
            raw_filename="too_large.txt",
        )
