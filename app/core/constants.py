"""Domain and infrastructure constants for the LexiRAG backend."""

from typing import Final

# Nebius Token Factory Defaults
DEFAULT_NEBIUS_BASE_URL: Final[str] = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_NEMOTRON_NANO_MODEL: Final[str] = "nvidia/nemotron-3-nano-30b-a3b"
DEFAULT_NEMOTRON_SUPER_MODEL: Final[str] = "nvidia/nemotron-3-super-120b-a12b"
DEFAULT_EMBEDDING_MODEL: Final[str] = "BAAI/bge-m3"

# Retrieval Hyperparameters
DEFAULT_TOP_K_RETRIEVAL: Final[int] = 5
DEFAULT_MIN_SIMILARITY_SCORE: Final[float] = 0.40
DEFAULT_TOP_K_SIMPLE: Final[int] = 3
DEFAULT_TOP_K_COMPLEX: Final[int] = 7
DEFAULT_MIN_SCORE_SIMPLE: Final[float] = 0.50
DEFAULT_MIN_SCORE_COMPLEX: Final[float] = 0.45
DEFAULT_SCORE_MARGIN_RATIO: Final[float] = 0.75
DEFAULT_MAX_CONTEXT_CHARS: Final[int] = 12000
DEFAULT_ENABLE_NEIGHBOR_EXPANSION: Final[bool] = True
DEFAULT_MAX_EXPANSION_DEPTH: Final[int] = 1
DEFAULT_MAX_NEIGHBORS_PER_CHUNK: Final[int] = 2
DEFAULT_MAX_PRIMARY_CHUNKS_TO_EXPAND: Final[int] = 2
DEFAULT_MAX_TOTAL_EXPANDED_CHUNKS: Final[int] = 3
DEFAULT_ENABLE_CITATION_VERIFICATION: Final[bool] = True
DEFAULT_ENABLE_SELECTIVE_LLM_VERIFIER: Final[bool] = False
DEFAULT_VERIFICATION_MIN_OVERLAP_RATIO: Final[float] = 0.40
DEFAULT_TOP_K_PER_DOCUMENT: Final[int] = 4
MAX_COMPARISON_DOCUMENTS: Final[int] = 5
DEFAULT_ENABLE_CROSS_DOC_RELATIONS: Final[bool] = True
MAX_QUERY_CHARACTER_LENGTH: Final[int] = 2000
MIN_QUERY_CHARACTER_LENGTH: Final[int] = 5

# Network Timeouts (seconds)
NEBIUS_REQUEST_TIMEOUT_SECONDS: Final[float] = 60.0
QDRANT_REQUEST_TIMEOUT_SECONDS: Final[float] = 15.0

# Retry Parameters for Nebius Token Factory
MAX_RETRY_ATTEMPTS: Final[int] = 3
INITIAL_BACKOFF_SECONDS: Final[float] = 1.0
MAX_BACKOFF_SECONDS: Final[float] = 8.0

# Supported Indian Legal Domains
LEGAL_DOMAIN_TAXATION: Final[str] = "taxation"
LEGAL_DOMAIN_CORPORATE: Final[str] = "corporate_law"
LEGAL_DOMAIN_EMPLOYMENT: Final[str] = "employment_labor"
LEGAL_DOMAIN_STARTUP: Final[str] = "startup_compliance"
LEGAL_DOMAIN_CRIMINAL: Final[str] = "criminal_procedure"
LEGAL_DOMAIN_GENERAL: Final[str] = "general_statutory"

VALID_LEGAL_DOMAINS: Final[tuple[str, ...]] = (
    LEGAL_DOMAIN_TAXATION,
    LEGAL_DOMAIN_CORPORATE,
    LEGAL_DOMAIN_EMPLOYMENT,
    LEGAL_DOMAIN_STARTUP,
    LEGAL_DOMAIN_CRIMINAL,
    LEGAL_DOMAIN_GENERAL,
)

# Document Ingestion & Chunking Constants
MAX_DOCUMENT_UPLOAD_SIZE_BYTES: Final[int] = 25 * 1024 * 1024  # 25 MB
ALLOWED_DOCUMENT_EXTENSIONS: Final[tuple[str, ...]] = ("pdf", "docx", "txt")
DEFAULT_CHUNK_SIZE_CHARS: Final[int] = 1000
DEFAULT_CHUNK_OVERLAP_CHARS: Final[int] = 150
DEFAULT_EMBEDDING_BATCH_SIZE: Final[int] = 16
QDRANT_VECTOR_DIMENSION: Final[int] = 1024  # Matches BAAI/bge-m3 dense vector shape

