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
