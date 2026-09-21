"""
Attachment Extractor: Identifies and extracts embedded files and PDF portfolio members.
Maintains clear parent-child document relationships without blindly merging content.
"""

from pathlib import Path
from typing import Dict, List, Any
import fitz  # PyMuPDF
from utils.logger import get_logger

logger = get_logger("attachment_extractor")


def extract_embedded_attachments(file_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
    """
    Extracts embedded attachments from a PDF.
    Returns:
    [
        {
            "name": filename,
            "size": int,
            "path": Path,
            "is_pdf": bool
        }
    ]
    """
    attachments = []
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        doc = fitz.open(file_path)
        emb_names = doc.embfile_names()

        for name in emb_names:
            try:
                emb_bytes = doc.embfile_get(name)
                dest_path = output_dir / f"embedded_{name}"
                dest_path.write_bytes(emb_bytes)

                attachments.append({
                    "name": name,
                    "size": len(emb_bytes),
                    "path": dest_path,
                    "is_pdf": name.lower().endswith(".pdf"),
                })
            except Exception as e:
                logger.warning(f"Could not extract embedded file {name}: {e}")

        doc.close()
    except Exception as e:
        logger.error(f"Error accessing embedded attachments in {file_path}: {e}")

    return attachments
