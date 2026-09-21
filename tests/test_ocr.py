"""
Unit tests for OCR Engine and Image Preprocessor.
"""

from pathlib import Path
from PIL import Image, ImageDraw
import pytest
from ingestion.ocr_engine import preprocess_image_for_ocr, ocr_page
from config import settings

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_image_preprocessing():
    # Create simple image
    img = Image.new("RGB", (200, 200), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "TEST OCR", fill=(0, 0, 0))

    enhanced = preprocess_image_for_ocr(img)
    assert enhanced is not None
    assert enhanced.size == (200, 200)


def test_ocr_on_scanned_pdf():
    pdf_path = FIXTURES_DIR / "scanned_car_policy.pdf"
    assert pdf_path.exists()

    if not settings.TESSERACT_CMD:
        pytest.skip("Tesseract executable not available on host.")

    result = ocr_page(pdf_path, 1)
    assert result["success"] is True
    assert result["char_count"] > 20
    text_upper = result["text"].upper()
    assert "TATA ACE" in text_upper or "GJ" in text_upper or "POLICY" in text_upper
