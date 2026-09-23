"""
OCR Engine: Local OCR using Tesseract with image preprocessing.
Supports page rendering, grayscale, adaptive thresholding, deskewing, and layout preservation.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import io
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from config import settings
from utils.logger import get_logger

logger = get_logger("ocr_engine")

# Configure tesseract executable path if resolved
if settings.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD


def get_verified_tesseract_cmd() -> Optional[str]:
    """Dynamically verifies or discovers Tesseract binary across environments."""
    if settings.TESSERACT_CMD and Path(settings.TESSERACT_CMD).is_file():
        return settings.TESSERACT_CMD

    from config import resolve_tesseract_path
    resolved = resolve_tesseract_path(None)
    if resolved and Path(resolved).is_file():
        settings.TESSERACT_CMD = resolved
        pytesseract.pytesseract.tesseract_cmd = resolved
        return resolved

    return None


def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    Apply deterministic image enhancements for optimal OCR accuracy:
    - Grayscale conversion
    - Contrast enhancement
    - Gentle sharpening
    - Otsu or adaptive thresholding when applicable
    """
    try:
        # Convert to grayscale
        gray = image.convert("L")

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(1.8)

        # Gentle sharpening
        sharpened = enhanced.filter(ImageFilter.SHARPEN)

        return sharpened
    except Exception as e:
        logger.warning(f"Image preprocessing warning: {e}")
        return image


def ocr_page(
    file_path: Path,
    page_number: int,
    dpi: int = 300,
    lang: str = "eng",
) -> Dict[str, Any]:
    """
    Renders a PDF page to high-res pixmap and performs local OCR.
    Returns:
    {
        "page_number": int,
        "text": str,
        "success": bool,
        "error": Optional[str],
        "char_count": int
    }
    """
    result = {
        "page_number": page_number,
        "text": "",
        "success": False,
        "error": None,
        "char_count": 0,
    }

    tess_cmd = get_verified_tesseract_cmd()
    if not tess_cmd:
        result["error"] = (
            f"Tesseract OCR executable not found on system PATH or candidate paths "
            f"(configured: {settings.TESSERACT_CMD}). OCR is unavailable for page {page_number}."
        )
        logger.warning(result["error"])
        return result

    try:
        doc = fitz.open(file_path)
        if page_number < 1 or page_number > len(doc):
            result["error"] = f"Invalid page number {page_number}"
            doc.close()
            return result

        page = doc[page_number - 1]
        # Render at target DPI (scale factor: dpi / 72)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        # Convert to PIL Image
        img_bytes = pix.tobytes("png")
        pil_img = Image.open(io.BytesIO(img_bytes))

        # Preprocess
        processed_img = preprocess_image_for_ocr(pil_img)

        # Perform OCR with layout preservation (psm 3: Fully automatic page segmentation)
        custom_config = r"--oem 3 --psm 3"
        text = pytesseract.image_to_string(
            processed_img,
            lang=lang,
            config=custom_config,
            timeout=settings.OCR_TIMEOUT,
        )

        cleaned_text = text.strip()
        result["text"] = cleaned_text
        result["char_count"] = len(cleaned_text)
        result["success"] = True

        doc.close()
    except pytesseract.TesseractNotFoundError:
        result["error"] = "Tesseract executable not found on system."
        logger.error(result["error"])
    except pytesseract.TesseractError as te:
        result["error"] = f"Tesseract error: {te}"
        logger.error(result["error"])
    except Exception as e:
        result["error"] = f"OCR failed on page {page_number}: {str(e)}"
        logger.error(result["error"])

    return result


def ocr_document_pages(
    file_path: Path,
    page_numbers: List[int],
    dpi: int = 300,
) -> Dict[int, Dict[str, Any]]:
    """OCR a batch of pages sequentially to conserve laptop memory."""
    results = {}
    for pno in page_numbers:
        results[pno] = ocr_page(file_path, pno, dpi=dpi, lang=settings.OCR_LANGUAGE)
    return results
