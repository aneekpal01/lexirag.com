"""HTTP middleware components including request correlation and tracing."""

import re
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import request_id_ctx

# Valid request ID pattern: alphanumeric and hyphens, 8-128 characters
REQUEST_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]{8,128}$")


class RequestCorrelationMiddleware(BaseHTTPMiddleware):
    """
    Ensures every HTTP interaction carries a traceable correlation ID (X-Request-ID).
    
    Behavior:
    1. Preserves valid client-supplied X-Request-ID headers.
    2. Generates a cryptographically random UUID4 if missing or invalid.
    3. Injects the ID into contextvars for structured logging across pipeline nodes.
    4. Attaches the X-Request-ID header to every outbound HTTP response.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        incoming_id = request.headers.get("X-Request-ID")
        
        if incoming_id and REQUEST_ID_PATTERN.match(incoming_id.strip()):
            correlation_id = incoming_id.strip()
        else:
            correlation_id = str(uuid.uuid4())

        # Store in request state and contextvar
        request.state.request_id = correlation_id
        token = request_id_ctx.set(correlation_id)

        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = correlation_id
            return response
        finally:
            request_id_ctx.reset(token)
