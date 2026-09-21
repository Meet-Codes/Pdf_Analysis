"""
Text Extractor: High-fidelity native text extraction with layout and reading order.
"""

from pathlib import Path
from typing import Dict, List, Any
import fitz  # PyMuPDF
from utils.logger import get_logger

logger = get_logger("text_extractor")


def extract_native_text(file_path: Path, page_numbers: List[int] = None) -> Dict[int, Dict[str, Any]]:
    """
    Extracts native text, blocks, and words from PDF pages using PyMuPDF.
    Returns dictionary keyed by 1-based page number:
    {
        page_num: {
            "text": "...",
            "blocks": [...],
            "word_count": int
        }
    }
    """
    results = {}
    try:
        doc = fitz.open(file_path)
        total_pages = len(doc)
        target_pages = page_numbers if page_numbers else list(range(1, total_pages + 1))

        for pno in target_pages:
            if 1 <= pno <= total_pages:
                page = doc[pno - 1]
                text = page.get_text("text")
                # Also get blocks for structural layout awareness
                raw_blocks = page.get_text("blocks")
                blocks = []
                for b in raw_blocks:
                    # b: (x0, y0, x1, y1, text, block_no, block_type)
                    if len(b) >= 5 and b[4].strip():
                        blocks.append({
                            "bbox": (b[0], b[1], b[2], b[3]),
                            "text": b[4].strip(),
                            "block_no": b[5] if len(b) > 5 else 0,
                        })

                results[pno] = {
                    "text": text.strip(),
                    "blocks": blocks,
                    "word_count": len(text.split()),
                }
        doc.close()
    except Exception as e:
        logger.error(f"Error extracting native text from {file_path}: {e}")

    return results
