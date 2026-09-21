"""
Production logger for Document Intelligence Platform.
Provides structured, non-leaking logging with file and console handlers.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from config import settings

_logger = None


def get_logger(name: str = "document_intelligence") -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger.getChild(name)

    logger = logging.getLogger("document_intelligence")
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Prevent duplicate handlers if reloaded by Streamlit
    if not logger.handlers:
        # Formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        logger.addHandler(console_handler)

        # Rotating File Handler
        try:
            settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                filename=settings.LOGS_DIR / "app.log",
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)
        except Exception as e:
            console_handler.setLevel(logging.DEBUG)
            logger.warning(f"Could not initialize file logger: {e}")

    _logger = logger
    return logger.getChild(name)
