"""Integration tests for FastAPI HTTP routes and exception handlers."""

from unittest.mock import AsyncMock, MagicMock
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_nebius_client, get_qdrant_wrapper
from app.clients.nebius import NebiusTokenFactoryClient
from app.clients.qdrant import QdrantClientWrapper, RetrievedStatutoryChunk
from app.core.exceptions import NebiusRateLimitError, QdrantServiceError
from app.main import create_application


@pytest.fixture
def test_client_and_mocks():
    app = create_application()

    mock_nebius = MagicMock(spec=NebiusTokenFactoryClient)
    mock_nebius.create_chat_completion = AsyncMock(
        side_effect=[
            "CLASSIFICATION: SIMPLE\nRATIONALE: Direct compliance question.",
            "Under Section 135 of the Companies Act 2013, every company meeting the net worth threshold must constitute a CSR committee...",
        ]
    )
    mock_nebius.create_embedding = AsyncMock(return_value=[0.01] * 1024)

    mock_qdrant = MagicMock(spec=QdrantClientWrapper)
    sample_chunk = RetrievedStatutoryChunk(
        chunk_id="csr-chunk-1",
        act_name="Companies Act, 2013",
        section="Section 135",
        sub_section="(1)",
        title="Corporate Social Responsibility",
        court_or_authority=None,
        citation_ref=None,
        content="Every company having net worth of rupees five hundred crore or more...",
        domain="corporate_law",
        similarity_score=0.92,
    )
    mock_qdrant.search_statutes = AsyncMock(return_value=[sample_chunk])
    mock_qdrant.check_health = AsyncMock(return_value=True)

    app.dependency_overrides[get_nebius_client] = lambda: mock_nebius
    app.dependency_overrides[get_qdrant_wrapper] = lambda: mock_qdrant

    client = TestClient(app)
    return client, mock_nebius, mock_qdrant


def test_health_check(test_client_and_mocks):
    client, _, _ = test_client_and_mocks
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "nebius_token_factory" in data
    assert data["qdrant"]["reachable"] is True


def test_query_unauthorized_missing_bearer(test_client_and_mocks):
    client, _, _ = test_client_and_mocks
    payload = {"query": "What are the rules for annual filing under ROC?"}
    response = client.post("/query", json=payload)
    assert response.status_code == 401
    assert response.json()["error_code"] == "AUTHENTICATION_FAILED"


def test_query_successful_execution(test_client_and_mocks):
    client, _, _ = test_client_and_mocks
    headers = {"Authorization": "Bearer dev-test-token"}
    payload = {
        "query": "What is the CSR threshold and committee requirement under Companies Act 2013?",
        "domain": "corporate_law",
    }
    response = client.post("/query", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "Section 135" in data["answer"]
    assert len(data["citations"]) == 1
    assert data["citations"][0]["act_name"] == "Companies Act, 2013"
    assert data["citations"][0]["section"] == "Section 135"
    assert data["complexity"] == "simple"
    assert data["execution_time_ms"] > 0
    assert data["retrieved_chunks_count"] == 1


def test_query_validation_error_too_short(test_client_and_mocks):
    client, _, _ = test_client_and_mocks
    headers = {"Authorization": "Bearer dev-test-token"}
    payload = {"query": "law"}  # Less than 5 characters
    response = client.post("/query", json=payload, headers=headers)
    assert response.status_code == 422


def test_nebius_rate_limit_error_handling(test_client_and_mocks):
    client, mock_nebius, _ = test_client_and_mocks
    mock_nebius.create_chat_completion = AsyncMock(
        side_effect=NebiusRateLimitError("Token Factory 429 quota exhausted", retry_after=10)
    )

    headers = {"Authorization": "Bearer dev-test-token"}
    payload = {"query": "What are the transfer pricing documentation rules?"}
    response = client.post("/query", json=payload, headers=headers)

    assert response.status_code == 429
    assert response.headers.get("Retry-After") == "10"
    data = response.json()
    assert data["error_code"] == "NEBIUS_RATE_LIMITED"


def test_qdrant_service_error_handling(test_client_and_mocks):
    client, _, mock_qdrant = test_client_and_mocks
    mock_qdrant.search_statutes = AsyncMock(
        side_effect=QdrantServiceError("Failed to reach Qdrant Cloud cluster")
    )

    headers = {"Authorization": "Bearer dev-test-token"}
    payload = {"query": "What is Section 9 IBC operational debt?"}
    response = client.post("/query", json=payload, headers=headers)

    assert response.status_code == 503
    data = response.json()
    assert data["error_code"] == "QDRANT_UNAVAILABLE"


def test_request_id_generated_when_missing(test_client_and_mocks):
    client, _, _ = test_client_and_mocks
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    request_id = response.headers.get("X-Request-ID")
    assert request_id is not None
    assert len(request_id) >= 16


def test_incoming_request_id_preserved(test_client_and_mocks):
    client, _, _ = test_client_and_mocks
    custom_id = "req-lexirag-trace-998877"
    response = client.get("/api/v1/health", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_id


def test_cors_trusted_origin_accepted(test_client_and_mocks):
    client, _, _ = test_client_and_mocks
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
    }
    response = client.options("/api/v1/health", headers=headers)
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert response.headers.get("access-control-allow-credentials") == "true"


def test_cors_untrusted_origin_rejected(test_client_and_mocks):
    client, _, _ = test_client_and_mocks
    headers = {
        "Origin": "http://malicious-adversary.com",
        "Access-Control-Request-Method": "POST",
    }
    response = client.options("/api/v1/health", headers=headers)
    # Origin should not be allowed
    assert response.headers.get("access-control-allow-origin") is None
