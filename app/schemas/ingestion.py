"""Request and response schemas for document ingestion and corpus management."""

from datetime import datetime, timezone
from typing import Any, Optional
from pydantic import BaseModel, Field


class DocumentChunkSummary(BaseModel):
    """Brief metadata summary for an indexed chunk."""

    chunk_id: str
    chunk_index: int
    section: str
    page_number: Optional[int] = None
    heading: Optional[str] = None
    token_count: int


class DocumentUploadResponse(BaseModel):
    """Receipt returned upon successful document ingestion and indexing."""

    document_id: str = Field(..., description="Canonical SHA-256 hash identifying the document.")
    filename: str = Field(..., description="Sanitized original document filename.")
    file_type: str = Field(..., description="Detected format ('pdf', 'docx', 'txt').")
    file_size_bytes: int = Field(..., description="Uploaded file size in bytes.")
    total_chunks: int = Field(..., ge=0, description="Total structure-aware chunks indexed.")
    detected_sections: list[str] = Field(default_factory=list, description="Unique legal sections surfaced.")
    status: str = Field(..., description="Status indicator: 'indexed' or 'already_indexed'.")
    message: str = Field(..., description="Human-readable status confirmation.")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ingestion timestamp.",
    )


class DocumentStatusResponse(BaseModel):
    """Detailed metadata and chunk breakdown for an existing indexed document."""

    document_id: str
    filename: Optional[str] = None
    total_chunks: int
    status: str = Field(..., description="'indexed' or 'not_found'")
    sample_chunks: list[dict[str, Any]] = Field(default_factory=list)


class DocumentDeleteResponse(BaseModel):
    """Confirmation payload returned when a document's vectors are purged."""

    document_id: str
    deleted: bool
    message: str
