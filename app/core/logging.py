"""Centralized logging configuration for LexiRAG."""

import logging
import sys


def setup_application_logging(log_level: str = "INFO") -> None:
    """Configures structured console logging across all service modules."""
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
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
        root_logger.addHandler(console_handler)

    # Quiet external verbose libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.INFO)


def get_logger(module_name: str) -> logging.Logger:
    """Returns a named logger for the calling module."""
    return logging.getLogger(module_name)
