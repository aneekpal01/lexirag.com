"""Centralized logging configuration for LexiRAG."""

from contextvars import ContextVar
import logging
import sys

# Global request ID context variable for coroutine-safe tracing
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


def get_current_request_id() -> str:
    """Returns the correlation ID of the active HTTP request, or '-' if outside request scope."""
    return request_id_ctx.get()


class RequestIdLogFilter(logging.Filter):
    """Injects the active request correlation ID into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_current_request_id()
        return True


def setup_application_logging(log_level: str = "INFO") -> None:
    """Configures structured console logging across all service modules."""
    log_format = (
        "%(asctime)s | %(levelname)-8s | [%(request_id)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    )
    date_format = "%Y-%m-%dT%H:%M:%S"

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers on reinitialization
    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(RequestIdLogFilter())
        root_logger.addHandler(console_handler)
    else:
        for handler in root_logger.handlers:
            handler.addFilter(RequestIdLogFilter())

    # Quiet external verbose libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.INFO)


def get_logger(module_name: str) -> logging.Logger:
    """Returns a named logger for the calling module."""
    return logging.getLogger(module_name)
