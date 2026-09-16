"""Standardized error response schemas."""

from datetime import datetime, timezone
from typing import Any, Optional
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standardized JSON payload returned for non-2xx HTTP responses."""

    error_code: str = Field(
        ...,
        description="Machine-readable error discriminator code.",
        examples=["NEBIUS_RATE_LIMITED", "QDRANT_UNAVAILABLE", "UNAUTHORIZED"],
    )
    message: str = Field(
        ...,
        description="Human-readable explanation of the error condition.",
    )
    details: Optional[dict[str, Any]] = Field(
        default=None,
        description="Supplemental debug or diagnostic context.",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC timestamp when the error occurred.",
    )
