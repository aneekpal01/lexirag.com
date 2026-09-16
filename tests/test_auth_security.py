"""Dedicated security test suite validating Clerk JWT authentication and production controls."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import jwt
from jwt import PyJWKClientError
import pytest
from pydantic import ValidationError

from app.auth.clerk import AuthenticatedUser, verify_clerk_token
from app.core.config import Settings
from app.core.exceptions import AuthenticationError


@pytest.fixture(scope="module")
def rsa_key_pair():
    """Generates an RSA key pair for cryptographic JWT testing."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")

    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")

    return private_pem, public_pem


def test_production_rejects_dev_mode_setting():
    """Verify Settings model validator strictly forbids clerk_dev_mode in production."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            environment="production",
            clerk_dev_mode=True,
            clerk_issuer_url="https://clerk.lexirag.in",
        )
    assert "clerk_dev_mode" in str(exc_info.value)
    assert "Security violation" in str(exc_info.value)


def test_production_accepts_secure_setting():
    """Verify production settings instantiate cleanly when clerk_dev_mode is False."""
    settings = Settings(
        environment="production",
        clerk_dev_mode=False,
        clerk_issuer_url="https://clerk.lexirag.in",
    )
    assert settings.environment == "production"
    assert settings.clerk_dev_mode is False


@pytest.mark.asyncio
async def test_production_rejects_synthetic_dev_token():
    """Verify synthetic dev tokens are rejected when running in production mode."""
    settings = Settings(
        environment="production",
        clerk_dev_mode=False,
        clerk_issuer_url="https://clerk.lexirag.in",
        clerk_pem_public_key="some-dummy-pem",
    )

    mock_credentials = MagicMock()
    mock_credentials.credentials = "dev-test-token"

    with pytest.raises(AuthenticationError) as exc_info:
        await verify_clerk_token(credentials=mock_credentials, settings=settings)

    assert "Invalid or unverified Clerk authentication token" in str(exc_info.value)


@pytest.mark.asyncio
async def test_production_rejects_unsigned_jwt():
    """Verify forged, unsigned tokens (none algorithm or verify_signature=False exploit) fail."""
    settings = Settings(
        environment="production",
        clerk_dev_mode=False,
        clerk_issuer_url="https://clerk.lexirag.in",
        clerk_pem_public_key="some-dummy-pem",
    )

    # Forged unsigned token with admin sub
    unsigned_token = jwt.encode({"sub": "attacker_admin"}, key="", algorithm="none")

    mock_credentials = MagicMock()
    mock_credentials.credentials = unsigned_token

    with pytest.raises(AuthenticationError):
        await verify_clerk_token(credentials=mock_credentials, settings=settings)


@pytest.mark.asyncio
async def test_valid_signed_jwt_verified_with_pem(rsa_key_pair):
    """Verify cryptographic signature verification succeeds with valid Clerk public PEM."""
    private_pem, public_pem = rsa_key_pair
    issuer_url = "https://clerk.lexirag.in"

    payload = {
        "sub": "user_production_advocate_001",
        "sid": "sess_live_9999",
        "email": "counsel@supremecourt.in",
        "role": "senior_advocate",
        "iss": issuer_url,
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
    }
    signed_token = jwt.encode(payload, key=private_pem, algorithm="RS256")

    settings = Settings(
        environment="production",
        clerk_dev_mode=False,
        clerk_issuer_url=issuer_url,
        clerk_pem_public_key=public_pem,
    )

    mock_credentials = MagicMock()
    mock_credentials.credentials = signed_token

    user = await verify_clerk_token(credentials=mock_credentials, settings=settings)
    assert isinstance(user, AuthenticatedUser)
    assert user.user_id == "user_production_advocate_001"
    assert user.email == "counsel@supremecourt.in"
    assert user.role == "senior_advocate"


@pytest.mark.asyncio
async def test_expired_signed_jwt_rejected(rsa_key_pair):
    """Verify expired signatures are strictly rejected."""
    private_pem, public_pem = rsa_key_pair
    issuer_url = "https://clerk.lexirag.in"

    payload = {
        "sub": "user_expired",
        "iss": issuer_url,
        "exp": datetime.now(timezone.utc) - timedelta(minutes=5),  # Expired
    }
    expired_token = jwt.encode(payload, key=private_pem, algorithm="RS256")

    settings = Settings(
        environment="production",
        clerk_dev_mode=False,
        clerk_issuer_url=issuer_url,
        clerk_pem_public_key=public_pem,
    )

    mock_credentials = MagicMock()
    mock_credentials.credentials = expired_token

    with pytest.raises(AuthenticationError) as exc_info:
        await verify_clerk_token(credentials=mock_credentials, settings=settings)

    assert "expired" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_jwks_dynamic_verification_success(rsa_key_pair):
    """Verify dynamic key resolution via Clerk JWKS client."""
    private_pem, public_pem = rsa_key_pair
    issuer_url = "https://clerk.lexirag.in"

    payload = {
        "sub": "user_jwks_verified",
        "sid": "sess_jwks_123",
        "email": "lawyer@lexirag.in",
        "iss": issuer_url,
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
    }
    signed_token = jwt.encode(payload, key=private_pem, algorithm="RS256", headers={"kid": "key_1"})

    settings = Settings(
        environment="production",
        clerk_dev_mode=False,
        clerk_issuer_url=issuer_url,
        clerk_pem_public_key="",  # Empty, triggers JWKS resolution
    )

    mock_credentials = MagicMock()
    mock_credentials.credentials = signed_token

    mock_signing_key = MagicMock()
    mock_signing_key.key = public_pem

    mock_jwks_client = MagicMock()
    mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key

    with patch("app.auth.clerk.get_jwks_client", return_value=mock_jwks_client):
        user = await verify_clerk_token(credentials=mock_credentials, settings=settings)

    assert user.user_id == "user_jwks_verified"
    assert user.email == "lawyer@lexirag.in"
    mock_jwks_client.get_signing_key_from_jwt.assert_called_once_with(signed_token)
