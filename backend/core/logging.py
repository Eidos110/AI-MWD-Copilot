"""Structured logging configuration."""

import logging
import sys
from backend.core.config import settings


def setup_logging() -> logging.Logger:
    """Configure application logging."""
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("mwd_copilot")
    return logger


logger = setup_logging()
