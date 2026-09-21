"""
Image Processor: Handles embedded images, photographs, signatures, and stamps.
Distinguishes meaningful text-bearing images from decorative graphics.
"""

from pathlib import Path
from typing import Dict, List, Any
import io
import fitz
from PIL import Image
import pytesseract
from config import settings
from utils.logger import get_logger

logger = get_logger("image_processor")


def extract_embedded_images(
    file_path: Path,
    page_number: int,
    min_dimension: int = 80,
) -> List[Dict[str, Any]]:
    """
    Extracts embedded images from a specific page.
    Filters out decorative icons and tiny artifacts (< min_dimension px).
    """
    extracted = []
    try:
        doc = fitz.open(file_path)
        if 1 <= page_number <= len(doc):
            page = doc[page_number - 1]
            image_list = page.get_images(full=True)

            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                width = base_image["width"]
                height = base_image["height"]

                # Filter decorative elements (tiny icons, 1px rules)
                if width < min_dimension or height < min_dimension:
                    continue

                pil_image = Image.open(io.BytesIO(image_bytes))
                extracted.append({
                    "image_index": img_idx,
                    "xref": xref,
                    "width": width,
                    "height": height,
                    "format": image_ext,
                    "pil_image": pil_image,
                    "is_large": width > 500 and height > 500,
                })

        doc.close()
    except Exception as e:
        logger.error(f"Error extracting embedded images from {file_path} page {page_number}: {e}")

    return extracted


def extract_text_from_image(pil_image: Image.Image) -> str:
    """OCR an isolated embedded image if it likely contains text."""
    if not settings.TESSERACT_CMD:
        return ""
    try:
        # Check aspect ratio / resolution
        w, h = pil_image.size
        if w < 100 or h < 30:
            return ""
        text = pytesseract.image_to_string(pil_image, config="--psm 6")
        return text.strip()
    except Exception:
        return ""
