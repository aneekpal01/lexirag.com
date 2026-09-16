"""HTTP API integration tests for document ingestion, status, and deletion endpoints."""

import io
from unittest.mock import AsyncMock, MagicMock
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_nebius_client, get_qdrant_wrapper
from app.clients.nebius import NebiusTokenFactoryClient
from app.clients.qdrant import QdrantClientWrapper
from app.main import create_application
from tests.test_ingestion_parsers import SAMPLE_PDF_BYTES


@pytest.fixture
def test_client_and_mocks():
    app = create_application()

    mock_nebius = MagicMock(spec=NebiusTokenFactoryClient)
    mock_nebius.create_embeddings_batch = AsyncMock(
        side_effect=lambda texts, batch_size=None: [[0.01] * 1024 for _ in texts]
    )

    mock_qdrant = MagicMock(spec=QdrantClientWrapper)
    mock_qdrant.ensure_collection_exists = AsyncMock(return_value=True)
    mock_qdrant.upsert_points = AsyncMock(return_value=None)
    mock_qdrant.get_document_chunks = AsyncMock(return_value=[])
    mock_qdrant.delete_document = AsyncMock(return_value=True)
    mock_qdrant.check_health = AsyncMock(return_value=True)

    app.dependency_overrides[get_nebius_client] = lambda: mock_nebius
    app.dependency_overrides[get_qdrant_wrapper] = lambda: mock_qdrant

    client = TestClient(app)
    return client, mock_nebius, mock_qdrant


def test_upload_document_unauthorized_missing_bearer(test_client_and_mocks):
    client, _, _ = test_client_and_mocks
    files = {"file": ("statute.pdf", io.BytesIO(SAMPLE_PDF_BYTES), "application/pdf")}
    response = client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 401
    assert response.json()["error_code"] == "AUTHENTICATION_FAILED"


def test_upload_document_success(test_client_and_mocks):
    client, _, _ = test_client_and_mocks
    headers = {"Authorization": "Bearer dev-test-token"}
    files = {"file": ("companies_act.pdf", io.BytesIO(SAMPLE_PDF_BYTES), "application/pdf")}
    data = {"domain": "corporate_law"}

    response = client.post("/api/v1/documents/upload", headers=headers, files=files, data=data)
    assert response.status_code == 201
    receipt = response.json()
    assert receipt["status"] == "indexed"
    assert receipt["filename"] == "companies_act.pdf"
    assert receipt["file_type"] == "pdf"
    assert receipt["total_chunks"] >= 1
    assert len(receipt["document_id"]) == 64


def test_upload_document_unsupported_extension_rejected(test_client_and_mocks):
    client, _, _ = test_client_and_mocks
    headers = {"Authorization": "Bearer dev-test-token"}
    files = {"file": ("script.exe", io.BytesIO(b"malicious_bytes"), "application/octet-stream")}

    response = client.post("/api/v1/documents/upload", headers=headers, files=files)
    assert response.status_code == 415
    assert "Unsupported file format" in response.json()["detail"]


def test_upload_document_empty_file_rejected(test_client_and_mocks):
    client, _, _ = test_client_and_mocks
    headers = {"Authorization": "Bearer dev-test-token"}
    files = {"file": ("empty.txt", io.BytesIO(b""), "text/plain")}

    response = client.post("/api/v1/documents/upload", headers=headers, files=files)
    assert response.status_code == 422


def test_get_document_status_success(test_client_and_mocks):
    client, _, mock_qdrant = test_client_and_mocks
    doc_id = "doc_test_123"
    mock_qdrant.get_document_chunks = AsyncMock(
        return_value=[
            {
                "chunk_id": "c-1",
                "chunk_index": 0,
                "section": "Section 185",
                "document_name": "companies_act.pdf",
                "token_count": 120,
            }
        ]
    )

    headers = {"Authorization": "Bearer dev-test-token"}
    response = client.get(f"/api/v1/documents/{doc_id}", headers=headers)
    assert response.status_code == 200
    status_data = response.json()
    assert status_data["document_id"] == doc_id
    assert status_data["status"] == "indexed"
    assert status_data["total_chunks"] == 1
    assert status_data["filename"] == "companies_act.pdf"


def test_get_document_status_not_found(test_client_and_mocks):
    client, _, mock_qdrant = test_client_and_mocks
    mock_qdrant.get_document_chunks = AsyncMock(return_value=[])

    headers = {"Authorization": "Bearer dev-test-token"}
    response = client.get("/api/v1/documents/non_existent_doc", headers=headers)
    assert response.status_code == 404


def test_delete_document_success(test_client_and_mocks):
    client, _, mock_qdrant = test_client_and_mocks
    doc_id = "doc_to_delete"
    mock_qdrant.get_document_chunks = AsyncMock(return_value=[{"chunk_id": "c-1"}])
    mock_qdrant.delete_document = AsyncMock(return_value=True)

    headers = {"Authorization": "Bearer dev-test-token"}
    response = client.delete(f"/api/v1/documents/{doc_id}", headers=headers)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["deleted"] is True
    mock_qdrant.delete_document.assert_called_once_with(document_id=doc_id)


def test_delete_document_not_found(test_client_and_mocks):
    client, _, mock_qdrant = test_client_and_mocks
    mock_qdrant.get_document_chunks = AsyncMock(return_value=[])

    headers = {"Authorization": "Bearer dev-test-token"}
    response = client.delete("/api/v1/documents/missing_doc", headers=headers)
    assert response.status_code == 404
