"""
Security and permission detection for PDF documents.
Detects encryption, password protection, print/copy restrictions, and malformed files.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import fitz  # PyMuPDF
from utils.logger import get_logger

logger = get_logger("security")


def check_pdf_security(file_path: Path) -> Dict[str, Any]:
    """
    Inspects PDF security attributes using PyMuPDF.
    Detects:
    - Encrypted status
    - Password protection / locks
    - Permissions: copy, print, modify, annotate
    Does not attempt to bypass encryption or crack passwords.
    """
    result = {
        "is_valid_pdf": False,
        "is_encrypted": False,
        "requires_password": False,
        "can_copy": True,
        "can_print": True,
        "can_modify": True,
        "can_annotate": True,
        "error_message": None,
    }

    if not file_path.exists():
        result["error_message"] = f"File does not exist: {file_path}"
        return result

    # Check magic header
    try:
        with open(file_path, "rb") as f:
            header = f.read(5)
            if header != b"%PDF-":
                result["error_message"] = "Invalid PDF signature. File may be corrupted or not a PDF."
                return result
    except Exception as e:
        result["error_message"] = f"Cannot read file header: {e}"
        return result

    try:
        doc = fitz.open(file_path)
        result["is_valid_pdf"] = True
        result["is_encrypted"] = doc.is_encrypted
        result["requires_password"] = doc.needs_pass

        if doc.needs_pass:
            result["error_message"] = "This document is password protected. Please provide an authorized unlocked copy or password."
            doc.close()
            return result

        # Check permissions flags
        # doc.permissions is an integer bitmask in PyMuPDF
        perms = doc.permissions
        result["can_print"] = bool(perms & fitz.PDF_PERM_PRINT)
        result["can_modify"] = bool(perms & fitz.PDF_PERM_MODIFY)
        result["can_copy"] = bool(perms & fitz.PDF_PERM_COPY)
        result["can_annotate"] = bool(perms & fitz.PDF_PERM_ANNOTATE)

        doc.close()
    except Exception as e:
        logger.warning(f"Error inspecting PDF security for {file_path}: {e}")
        result["error_message"] = f"Unable to parse document security structure: {str(e)}"

    return result
