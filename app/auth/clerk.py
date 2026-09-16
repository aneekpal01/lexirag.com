"""Clerk authentication integration and route protection dependency."""

from functools import lru_cache, wraps
from typing import Any, Callable, Optional
from fastapi import Depends, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
from jwt import PyJWKClient, PyJWKClientError
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.core.exceptions import AuthenticationError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Security scheme for OpenAPI documentation
bearer_security_scheme = HTTPBearer(auto_error=False)


@lru_cache(maxsize=4)
def get_jwks_client(jwks_url: str) -> PyJWKClient:
    """Thread-safe, cached JWKS client for Clerk public key discovery."""
    return PyJWKClient(jwks_url, cache_keys=True, max_cached_keys=16)


class AuthenticatedUser(BaseModel):
    """User context populated from verified Clerk JWT claims."""

    user_id: str
    session_id: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    raw_claims: dict[str, Any] = {}


async def verify_clerk_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_security_scheme),
    settings: Settings = Depends(get_settings),
) -> AuthenticatedUser:
    """
    FastAPI dependency cryptographically validating Clerk JWT session tokens.
    
    Security Standards:
    1. Dev mode bypass ONLY allowed in non-production environments with synthetic test tokens.
    2. In production, tokens MUST be cryptographically verified using Clerk PEM or Clerk JWKS.
    3. Unsigned or unverified tokens are strictly rejected.
    """
    if credentials is None:
        raise AuthenticationError("Authorization header with Bearer token is missing")

    token = credentials.credentials.strip()
    if not token:
        raise AuthenticationError("Bearer token string is empty")

    # Developer / Hackathon test bypass hook (strictly disabled in production)
    if settings.environment != "production" and settings.clerk_dev_mode:
        if token == "dev-test-token" or token.startswith("dev-user-"):
            logger.debug("Clerk dev-mode active: accepting synthetic test token")
            return AuthenticatedUser(
                user_id="user_lexirag_hackathon_demo",
                session_id="sess_demo_12345",
                email="partner@azbpartners.example",
                role="advocate",
                raw_claims={"sub": "user_lexirag_hackathon_demo", "azp": "lexirag-portal"},
            )

    # Cryptographic JWT Signature Verification (Zero unverified fallback)
    try:
        if settings.clerk_pem_public_key:
            # Verified using pre-configured Clerk public PEM
            decoded_claims = jwt.decode(
                token,
                key=settings.clerk_pem_public_key,
                algorithms=["RS256"],
                issuer=settings.clerk_issuer_url,
                options={"verify_aud": False, "verify_signature": True},
            )
        else:
            # Verified dynamically against Clerk JWKS endpoint
            jwks_url = f"{settings.clerk_issuer_url.rstrip('/')}/.well-known/jwks.json"
            jwks_client = get_jwks_client(jwks_url)
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            
            decoded_claims = jwt.decode(
                token,
                key=signing_key.key,
                algorithms=["RS256"],
                issuer=settings.clerk_issuer_url,
                options={"verify_aud": False, "verify_signature": True},
            )

        user_id = decoded_claims.get("sub")
        if not user_id:
            raise AuthenticationError("Clerk JWT missing subject ('sub') claim")

        return AuthenticatedUser(
            user_id=user_id,
            session_id=decoded_claims.get("sid"),
            email=decoded_claims.get("email"),
            role=decoded_claims.get("role", "legal_counsel"),
            raw_claims=decoded_claims,
        )

    except jwt.ExpiredSignatureError as exc:
        logger.warning("Clerk JWT expired: %s", exc)
        raise AuthenticationError("Clerk authentication token has expired") from exc
    except (jwt.InvalidTokenError, PyJWKClientError) as exc:
        logger.warning("Invalid Clerk JWT structure, signature, or key discovery failure: %s", exc)
        raise AuthenticationError("Invalid or unverified Clerk authentication token") from exc
    except AuthenticationError:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during Clerk token verification")
        raise AuthenticationError(f"Authentication validation failed: {exc}") from exc


def require_clerk_auth(func: Callable) -> Callable:
    """
    Decorator for non-FastAPI route handlers or background tasks requiring Clerk session validation.
    """
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        if "user" not in kwargs or not isinstance(kwargs["user"], AuthenticatedUser):
            raise AuthenticationError("Operation requires verified Clerk authenticated user context")
        return await func(*args, **kwargs)

    return wrapper
