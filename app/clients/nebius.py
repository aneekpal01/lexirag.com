"""Client wrapper for Nebius Token Factory serving NVIDIA Nemotron & BGE-M3 models."""

import time
from typing import Any, Optional
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import Settings
from app.core.constants import (
    INITIAL_BACKOFF_SECONDS,
    MAX_BACKOFF_SECONDS,
    MAX_RETRY_ATTEMPTS,
    NEBIUS_REQUEST_TIMEOUT_SECONDS,
)
from app.core.exceptions import (
    NebiusExtractionError,
    NebiusRateLimitError,
    NebiusServiceError,
    NebiusTimeoutError,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


def extract_reasoning_or_content(choice_message: Any) -> str:
    """
    Extracts the textual payload from a Nebius Token Factory Nemotron response.
    
    CRITICAL ARCHITECTURAL REQUIREMENT:
    NVIDIA Nemotron models deployed on Nebius Token Factory operate as reasoning models.
    They deliver their response trace inside the `reasoning_content` field rather than
    the standard OpenAI `content` field.
    
    This parser checks `reasoning_content` first across object attributes and dictionary
    representations, gracefully falling back to `content`, and raising a domain-level
    NebiusExtractionError if neither yields valid content.
    """
    if choice_message is None:
        raise NebiusExtractionError("Nebius response choice message is None")

    # 1. Attribute access (Pydantic / OpenAI ChatCompletionMessage)
    reasoning = getattr(choice_message, "reasoning_content", None)
    if reasoning and str(reasoning).strip():
        return str(reasoning).strip()

    # 2. Check OpenAI model_extra dictionary (Pydantic v2 dynamic extra fields)
    if hasattr(choice_message, "model_extra") and choice_message.model_extra:
        extra_reasoning = choice_message.model_extra.get("reasoning_content")
        if extra_reasoning and str(extra_reasoning).strip():
            return str(extra_reasoning).strip()

    # 3. Dictionary access if serialized or dict-like
    if isinstance(choice_message, dict):
        dict_reasoning = choice_message.get("reasoning_content")
        if dict_reasoning and str(dict_reasoning).strip():
            return str(dict_reasoning).strip()

    # 4. Standard content fallback
    standard_content = getattr(choice_message, "content", None)
    if standard_content and str(standard_content).strip():
        return str(standard_content).strip()

    if isinstance(choice_message, dict):
        dict_content = choice_message.get("content")
        if dict_content and str(dict_content).strip():
            return str(dict_content).strip()

    raise NebiusExtractionError(
        "Nebius completion payload contained neither 'reasoning_content' nor 'content'"
    )


class NebiusTokenFactoryClient:
    """Encapsulates all inference and embedding requests directed to Nebius Token Factory."""

    def __init__(self, settings: Settings, client: Optional[AsyncOpenAI] = None):
        self.settings = settings
        self.client = client or AsyncOpenAI(
            base_url=settings.nebius_base_url,
            api_key=settings.nebius_api_key,
            timeout=NEBIUS_REQUEST_TIMEOUT_SECONDS,
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=INITIAL_BACKOFF_SECONDS, max=MAX_BACKOFF_SECONDS),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
        before_sleep=before_sleep_log(logger, "WARNING"),
    )
    async def create_chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 3000,
    ) -> str:
        """
        Executes a chat completion via Nebius Token Factory and parses the reasoning content.
        
        Applies exponential backoff retries for transient 429 rate limits and socket timeouts.
        """
        start_time = time.perf_counter()
        logger.debug("Dispatching chat completion request to model: %s", model)

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except RateLimitError as exc:
            logger.error("Nebius Token Factory 429 Rate Limit exceeded for model %s: %s", model, exc)
            raise NebiusRateLimitError(
                message=f"Rate limit exceeded on Nebius Token Factory for model {model}",
                retry_after=5,
            ) from exc
        except APITimeoutError as exc:
            logger.error("Nebius Token Factory timeout on model %s after %ss", model, NEBIUS_REQUEST_TIMEOUT_SECONDS)
            raise NebiusTimeoutError(
                f"Inference request timed out on Nebius Token Factory for model {model}"
            ) from exc
        except APIConnectionError as exc:
            logger.error("Failed to connect to Nebius Token Factory base URL %s: %s", self.settings.nebius_base_url, exc)
            raise NebiusServiceError(
                f"Network connection failed when reaching Nebius Token Factory: {exc}"
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected error during Nebius chat completion on model %s", model)
            raise NebiusServiceError(f"Nebius chat completion failed: {exc}") from exc

        if not response.choices:
            raise NebiusExtractionError("Nebius response choices list is empty")

        choice = response.choices[0]
        extracted_text = extract_reasoning_or_content(choice.message)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Nebius chat completion succeeded | model: %s | elapsed: %.2fms | length: %d chars",
            model,
            elapsed_ms,
            len(extracted_text),
        )
        return extracted_text

    @retry(
        reraise=True,
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=INITIAL_BACKOFF_SECONDS, max=MAX_BACKOFF_SECONDS),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
        before_sleep=before_sleep_log(logger, "WARNING"),
    )
    async def create_embedding(self, text: str) -> list[float]:
        """
        Generates dense vector embeddings using BGE-M3 via Nebius Token Factory.
        """
        clean_text = text.strip().replace("\n", " ")
        if not clean_text:
            raise NebiusServiceError("Cannot generate embedding for empty string")

        logger.debug("Generating dense vector via Nebius embedding model: %s", self.settings.embedding_model)

        try:
            response = await self.client.embeddings.create(
                model=self.settings.embedding_model,
                input=clean_text,
            )
        except RateLimitError as exc:
            raise NebiusRateLimitError(
                message=f"Rate limit exceeded on Nebius embedding model {self.settings.embedding_model}"
            ) from exc
        except APITimeoutError as exc:
            raise NebiusTimeoutError("Nebius embedding generation timed out") from exc
        except APIConnectionError as exc:
            raise NebiusServiceError(f"Nebius embedding connection failed: {exc}") from exc
        except Exception as exc:
            raise NebiusServiceError(f"Nebius embedding generation failed: {exc}") from exc

        if not response.data or not response.data[0].embedding:
            raise NebiusServiceError("Nebius embedding endpoint returned empty vector payload")

        return response.data[0].embedding
