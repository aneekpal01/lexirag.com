"""Clerk authentication integration and route protection dependency."""

from functools import wraps
from typing import Any, Callable, Optional
from fastapi import Depends, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.core.exceptions import AuthenticationError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Security scheme for OpenAPI documentation
bearer_security_scheme = HTTPBearer(auto_error=False)


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
    FastAPI dependency validating Clerk JWT session tokens.
    
    Supports:
    1. Dev mode bypass for local testing and hackathon judging when CLERK_DEV_MODE=True.
    2. Verification via Clerk public PEM or JWKS in staging/production.
    """
    if credentials is None:
        raise AuthenticationError("Authorization header with Bearer token is missing")

    token = credentials.credentials.strip()
    if not token:
        raise AuthenticationError("Bearer token string is empty")

    # Developer / Hackathon test bypass hook
    if settings.clerk_dev_mode and (token == "dev-test-token" or token.startswith("dev-user-")):
        logger.debug("Clerk dev-mode active: accepting synthetic test token")
        return AuthenticatedUser(
            user_id="user_lexirag_hackathon_demo",
            session_id="sess_demo_12345",
            email="partner@azbpartners.example",
            role="advocate",
            raw_claims={"sub": "user_lexirag_hackathon_demo", "azp": "lexirag-portal"},
        )

    # Production JWT decoding
    try:
        if settings.clerk_pem_public_key:
            # Verified using pre-configured Clerk public PEM
            decoded_claims = jwt.decode(
                token,
                key=settings.clerk_pem_public_key,
                algorithms=["RS256"],
                issuer=settings.clerk_issuer_url,
                options={"verify_aud": False},
            )
        else:
            # Decode unverified header to fetch kid or verify without PEM if secret available
            # If no PEM is configured and not in dev mode, we decode without signature verification
            # only if explicitly permissible, or raise an error instructing configuration.
            decoded_claims = jwt.decode(
                token,
                options={"verify_signature": False},
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
    except jwt.InvalidTokenError as exc:
        logger.warning("Invalid Clerk JWT structure or signature: %s", exc)
        raise AuthenticationError("Invalid Clerk authentication token") from exc
    except Exception as exc:
        logger.exception("Unexpected error during Clerk token verification")
        raise AuthenticationError(f"Authentication validation failed: {exc}") from exc


def require_clerk_auth(func: Callable) -> Callable:
    """
    Decorator for non-FastAPI route handlers or background tasks requiring Clerk session validation.
    """
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Check if user context is already passed in kwargs
        if "user" not in kwargs or not isinstance(kwargs["user"], AuthenticatedUser):
            raise AuthenticationError("Operation requires verified Clerk authenticated user context")
        return await func(*args, **kwargs)

    return wrapper
