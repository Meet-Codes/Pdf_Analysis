"""
File handling utilities: unique ID generation, safe saving, and hashing.
"""

import hashlib
import time
import uuid
from pathlib import Path
from typing import Union
from config import settings


def generate_document_id(filename: str = "") -> str:
    """Generate a unique, chronological document ID."""
    timestamp = int(time.time())
    short_uuid = uuid.uuid4().hex[:8]
    return f"doc_{timestamp}_{short_uuid}"


def compute_sha256(filepath_or_bytes: Union[Path, bytes]) -> str:
    """Compute SHA256 checksum for audit and deduplication."""
    sha = hashlib.sha256()
    if isinstance(filepath_or_bytes, bytes):
        sha.update(filepath_or_bytes)
    else:
        with open(filepath_or_bytes, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
    return sha.hexdigest()


def save_uploaded_file(file_bytes: bytes, filename: str) -> Path:
    """
    Safely save uploaded PDF bytes into the configured upload directory.
    Uses sanitized name with unique prefix to avoid collision.
    """
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # Sanitize filename
    safe_name = Path(filename).name
    # Prepend short hash or uuid to avoid overwriting identical filenames
    file_hash = hashlib.md5(file_bytes[:1024]).hexdigest()[:6]
    target_filename = f"{file_hash}_{safe_name}"
    target_path = settings.UPLOAD_DIR / target_filename
    target_path.write_bytes(file_bytes)
    return target_path
