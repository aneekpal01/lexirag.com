"""Application configuration management using Pydantic Settings."""

from functools import lru_cache
from typing import Literal
from pydantic import Field, HttpUrl, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.constants import (
    DEFAULT_CHUNK_OVERLAP_CHARS,
    DEFAULT_CHUNK_SIZE_CHARS,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_ENABLE_CITATION_VERIFICATION,
    DEFAULT_ENABLE_NEIGHBOR_EXPANSION,
    DEFAULT_ENABLE_SELECTIVE_LLM_VERIFIER,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_EXPANSION_DEPTH,
    DEFAULT_MAX_NEIGHBORS_PER_CHUNK,
    DEFAULT_MAX_PRIMARY_CHUNKS_TO_EXPAND,
    DEFAULT_MAX_TOTAL_EXPANDED_CHUNKS,
    DEFAULT_MIN_SCORE_COMPLEX,
    DEFAULT_MIN_SCORE_SIMPLE,
    DEFAULT_MIN_SIMILARITY_SCORE,
    DEFAULT_NEBIUS_BASE_URL,
    DEFAULT_NEMOTRON_NANO_MODEL,
    DEFAULT_NEMOTRON_SUPER_MODEL,
    DEFAULT_SCORE_MARGIN_RATIO,
    DEFAULT_TOP_K_COMPLEX,
    DEFAULT_TOP_K_RETRIEVAL,
    DEFAULT_TOP_K_SIMPLE,
    DEFAULT_TOP_K_PER_DOCUMENT,
    DEFAULT_VERIFICATION_MIN_OVERLAP_RATIO,
    MAX_COMPARISON_DOCUMENTS,
    DEFAULT_ENABLE_CROSS_DOC_RELATIONS,
    MAX_DOCUMENT_UPLOAD_SIZE_BYTES,
)


class Settings(BaseSettings):
    """Runtime configuration validated at application startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Runtime Environment
    environment: Literal["development", "production", "testing"] = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Nebius Token Factory
    nebius_api_key: str = Field(
        default="mock-nebius-key",
        description="Nebius Token Factory API key for LLM inference and embeddings.",
    )
    nebius_base_url: str = Field(
        default=DEFAULT_NEBIUS_BASE_URL,
        description="OpenAI-compatible base URL for Nebius Token Factory.",
    )
    nemotron_nano_model: str = Field(
        default=DEFAULT_NEMOTRON_NANO_MODEL,
        description="Model identifier for rapid query classification and standard lookups.",
    )
    nemotron_super_model: str = Field(
        default=DEFAULT_NEMOTRON_SUPER_MODEL,
        description="Model identifier for nuanced Indian legal analysis.",
    )
    embedding_model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        description="Embedding model identifier served via Nebius Token Factory.",
    )

    # Qdrant Cloud Vector Database
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="URL for cloud-hosted or local Qdrant cluster.",
    )
    qdrant_api_key: str = Field(
        default="mock-qdrant-key",
        description="API Key for Qdrant Cloud authentication.",
    )
    qdrant_collection_name: str = Field(
        default="indian_legal_corpus",
        description="Qdrant collection containing pre-indexed Indian statutes and judgments.",
    )

    # Retrieval Configuration
    top_k_retrieval_limit: int = Field(
        default=DEFAULT_TOP_K_RETRIEVAL,
        ge=1,
        le=20,
        description="Maximum statutory chunks to retrieve per search pass.",
    )
    min_similarity_score: float = Field(
        default=DEFAULT_MIN_SIMILARITY_SCORE,
        ge=0.0,
        le=1.0,
        description="Cosine similarity floor for retrieved document chunks.",
    )
    top_k_simple: int = Field(
        default=DEFAULT_TOP_K_SIMPLE,
        ge=1,
        le=20,
        description="Adaptive top-k limit for simple definitional/procedural queries.",
    )
    top_k_complex: int = Field(
        default=DEFAULT_TOP_K_COMPLEX,
        ge=1,
        le=20,
        description="Adaptive top-k limit for complex multi-statute queries.",
    )
    min_score_simple: float = Field(
        default=DEFAULT_MIN_SCORE_SIMPLE,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score floor for simple queries.",
    )
    min_score_complex: float = Field(
        default=DEFAULT_MIN_SCORE_COMPLEX,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score floor for complex queries.",
    )
    score_margin_ratio: float = Field(
        default=DEFAULT_SCORE_MARGIN_RATIO,
        ge=0.1,
        le=1.0,
        description="Relative score pruning ratio (drops chunks below top_score * ratio).",
    )
    max_context_chars: int = Field(
        default=DEFAULT_MAX_CONTEXT_CHARS,
        ge=500,
        le=50000,
        description="Maximum total characters permitted in compiled legal context block.",
    )

    # Phase 3 Evidence Graph & Neighbor Expansion Configuration
    enable_neighbor_expansion: bool = Field(
        default=DEFAULT_ENABLE_NEIGHBOR_EXPANSION,
        description="Whether to perform controlled neighboring chunk expansion on retrieved evidence.",
    )
    max_expansion_depth: int = Field(
        default=DEFAULT_MAX_EXPANSION_DEPTH,
        ge=1,
        le=1,
        description="Maximum expansion depth from primary chunks (strictly 1 in Phase 3).",
    )
    max_neighbors_per_chunk: int = Field(
        default=DEFAULT_MAX_NEIGHBORS_PER_CHUNK,
        ge=1,
        le=2,
        description="Maximum neighbor chunks to retrieve per primary chunk (previous + next).",
    )
    max_primary_chunks_to_expand: int = Field(
        default=DEFAULT_MAX_PRIMARY_CHUNKS_TO_EXPAND,
        ge=1,
        le=5,
        description="Maximum number of top primary chunks eligible for neighbor expansion.",
    )
    max_total_expanded_chunks: int = Field(
        default=DEFAULT_MAX_TOTAL_EXPANDED_CHUNKS,
        ge=1,
        le=10,
        description="Maximum total supporting neighbor chunks permitted across the query.",
    )

    # Phase 4 Citation Verification & Evidence Attribution Configuration
    enable_citation_verification: bool = Field(
        default=DEFAULT_ENABLE_CITATION_VERIFICATION,
        description="Whether to execute claim extraction and citation verification against retrieved evidence.",
    )
    enable_selective_llm_verifier: bool = Field(
        default=DEFAULT_ENABLE_SELECTIVE_LLM_VERIFIER,
        description="Whether to run selective LLM verification as a fallback when deterministic verification is ambiguous.",
    )
    verification_min_overlap_ratio: float = Field(
        default=DEFAULT_VERIFICATION_MIN_OVERLAP_RATIO,
        ge=0.0,
        le=1.0,
        description="Minimum lexical/entity token overlap ratio heuristic for substantive support evaluation.",
    )

    # Phase 5 Multi-Document Research & Comparison Configuration
    top_k_per_document: int = Field(
        default=DEFAULT_TOP_K_PER_DOCUMENT,
        ge=1,
        le=10,
        description="Targeted retrieval chunk quota per document during multi-document research.",
    )
    max_comparison_documents: int = Field(
        default=MAX_COMPARISON_DOCUMENTS,
        ge=2,
        le=10,
        description="Maximum number of documents allowed in a single comparison query.",
    )
    enable_cross_document_relations: bool = Field(
        default=DEFAULT_ENABLE_CROSS_DOC_RELATIONS,
        description="Whether to run cross-document provision alignment and relation detection.",
    )

    # Clerk Authentication
    clerk_dev_mode: bool = Field(
        default=True,
        description="When True, allows test tokens or bypass for local development/testing.",
    )
    clerk_issuer_url: str = Field(
        default="https://clerk.lexirag.dev",
        description="Clerk instance Frontend API / Issuer URL.",
    )
    clerk_pem_public_key: str = Field(
        default="",
        description="Optional Clerk PEM public key for offline JWT verification.",
    )

    # CORS Configuration
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins for web frontend clients.",
    )

    # Document Ingestion & Chunking Configuration
    max_upload_size_bytes: int = Field(
        default=MAX_DOCUMENT_UPLOAD_SIZE_BYTES,
        description="Maximum inbound file size permitted for document ingestion (25 MB).",
    )
    chunk_size_chars: int = Field(
        default=DEFAULT_CHUNK_SIZE_CHARS,
        ge=200,
        le=4000,
        description="Target character size constraint per chunk.",
    )
    chunk_overlap_chars: int = Field(
        default=DEFAULT_CHUNK_OVERLAP_CHARS,
        ge=0,
        le=1000,
        description="Overlapping characters between contiguous document chunks.",
    )
    embedding_batch_size: int = Field(
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        ge=1,
        le=64,
        description="Batch size for concurrent array embeddings sent to Nebius Token Factory.",
    )

    @field_validator("nebius_base_url")
    @classmethod
    def ensure_trailing_slash(cls, value: str) -> str:
        """OpenAI client requires base_url to terminate with a slash."""
        if not value.endswith("/"):
            return f"{value}/"
        return value

    @model_validator(mode="after")
    def validate_production_security(self) -> "Settings":
        """Disallow insecure development overrides when deployed in production."""
        if self.environment == "production" and self.clerk_dev_mode:
            raise ValueError(
                "Security violation: 'clerk_dev_mode' cannot be enabled when 'environment' is 'production'."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Singleton getter for application settings."""
    return Settings()
