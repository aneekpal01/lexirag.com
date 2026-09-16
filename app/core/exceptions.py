"""Domain exceptions for LexiRAG backend services."""

from typing import Any, Optional


class LexiRAGException(Exception):
    """Base exception for all domain-specific errors in LexiRAG."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NebiusServiceError(LexiRAGException):
    """Raised when an interaction with Nebius Token Factory fails."""


class NebiusRateLimitError(NebiusServiceError):
    """Raised when Nebius Token Factory returns HTTP 429 Too Many Requests."""

    def __init__(self, message: str = "Nebius Token Factory rate limit exceeded", retry_after: Optional[int] = None):
        details = {"retry_after_seconds": retry_after} if retry_after is not None else {}
        super().__init__(message, details)
        self.retry_after = retry_after


class NebiusTimeoutError(NebiusServiceError):
    """Raised when an inference or embedding request times out."""


class NebiusExtractionError(NebiusServiceError):
    """Raised when a Nemotron response fails to yield valid text in reasoning_content or content."""


class QdrantServiceError(LexiRAGException):
    """Raised when communication with Qdrant Cloud fails."""


class MalformedQdrantPayloadError(QdrantServiceError):
    """Raised when a vector search result contains an unparseable or corrupted payload."""


class AuthenticationError(LexiRAGException):
    """Raised when Clerk authentication credentials cannot be validated."""


class InvalidLegalQueryError(LexiRAGException):
    """Raised when a query fails domain validation rules."""
