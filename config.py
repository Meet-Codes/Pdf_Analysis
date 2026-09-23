"""
Configuration management for Production Document Intelligence Platform.
Loads configuration from environment variables (.env) with production-grade defaults.
"""

import os
import shutil
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent

# Load .env file
load_dotenv(BASE_DIR / ".env")


def resolve_tesseract_path(configured_path: Optional[str]) -> Optional[str]:
    """
    Deterministically resolve Tesseract executable path on Windows / Linux / MacOS.
    Checks explicit configuration first, then PATH, then well-known installation paths.
    """
    if configured_path and Path(configured_path).is_file():
        return str(Path(configured_path).resolve())

    # Check if tesseract is available on system PATH
    which_path = shutil.which("tesseract")
    if which_path:
        return which_path

    # Common Windows, Linux and MacOS installation locations
    candidate_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        os.path.expandvars(r"%USERPROFILE%\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/usr/bin/tesseract-ocr",
        "/opt/homebrew/bin/tesseract",
    ]

    for candidate in candidate_paths:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())

    return None


class Settings:
    # Service URLs & Models
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:4b")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    LOCAL_EMBEDDING_FALLBACK: str = os.getenv("LOCAL_EMBEDDING_FALLBACK", "all-MiniLM-L6-v2")
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "30"))

    # RAG candidate pool sizes
    RETRIEVAL_CANDIDATE_POOL: int = int(os.getenv("RETRIEVAL_CANDIDATE_POOL", "18"))
    RETRIEVAL_FINAL_K: int = int(os.getenv("RETRIEVAL_FINAL_K", "4"))

    # OCR Settings
    TESSERACT_CMD: Optional[str] = resolve_tesseract_path(os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe"))
    OCR_LANGUAGE: str = os.getenv("OCR_LANGUAGE", "eng")
    OCR_DPI: int = int(os.getenv("OCR_DPI", "300"))
    OCR_TIMEOUT: int = int(os.getenv("OCR_TIMEOUT", "30"))

    # Storage Paths (all using pathlib.Path)
    BASE_DIR: Path = BASE_DIR
    UPLOAD_DIR: Path = BASE_DIR / os.getenv("UPLOAD_DIR", "data/uploads")
    PROCESSED_DIR: Path = BASE_DIR / os.getenv("PROCESSED_DIR", "data/processed")
    VECTOR_DB_PATH: Path = BASE_DIR / os.getenv("VECTOR_DB_PATH", "data/vector_db")
    CACHE_DIR: Path = BASE_DIR / os.getenv("CACHE_DIR", "data/cache")
    LOGS_DIR: Path = BASE_DIR / "logs"

    # Runtime and Limits
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    ENABLE_DEBUG_MODE: bool = os.getenv("ENABLE_DEBUG_MODE", "false").lower() in ("true", "1", "yes")

    # Hardware Optimization (Laptop 8GB RAM, RTX 3050 4GB VRAM)
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "2"))
    MAX_IMAGE_DIMENSION: int = int(os.getenv("MAX_IMAGE_DIMENSION", "2500"))

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        for path in [
            self.UPLOAD_DIR,
            self.PROCESSED_DIR,
            self.VECTOR_DB_PATH,
            self.CACHE_DIR,
            self.LOGS_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)


# Global settings singleton
settings = Settings()
settings.ensure_directories()
