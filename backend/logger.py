"""
Centralized logging configuration for DocMind AI backend.

Usage in any module:
    from backend.logger import get_logger
    logger = get_logger(__name__)
    
    logger.info("Document %s processed with %d chunks", doc_id, len(chunks))
    logger.warning("No results above threshold for query: %s", query)
    logger.error("OpenRouter API error: %s", str(e))
    logger.debug("FAISS search returned %d candidates", len(candidates))
"""

import logging
import sys
import os
from logging.handlers import RotatingFileHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "docmind.log")

# Root log level — change to DEBUG for verbose output
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

_fmt = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger for the given module name.
    
    Each call with the same name returns the same logger instance.
    Handlers are only added once to prevent duplicate log entries.
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger  # Already configured — avoid adding duplicate handlers
    
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    
    # Console handler (stdout) — always active
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(_fmt)
    logger.addHandler(console_handler)
    
    # Rotating file handler — max 5MB per file, keep 3 backups
    try:
        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(_fmt)
        logger.addHandler(file_handler)
    except (OSError, PermissionError):
        # File logging not critical — continue with console only
        pass
    
    # Prevent log messages from propagating to the root logger (avoids duplicates)
    logger.propagate = False
    
    return logger
