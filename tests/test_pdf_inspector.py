"""
Unit tests for PDF Inspector and Page Analyzer.
"""

from pathlib import Path
import pytest
from ingestion.pdf_inspector import inspect_pdf
from schemas.base import PageType

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_native_pdf_inspection():
    pdf_path = FIXTURES_DIR / "tata_ace_insurance.pdf"
    assert pdf_path.exists()

    result = inspect_pdf(pdf_path)
    assert result.page_count == 1
    assert result.is_encrypted is False
    assert result.requires_password is False
    assert result.can_copy is True
    assert "Native text PDF" in result.detected_pdf_categories
    assert len(result.pages) == 1

    p1 = result.pages[0]
    assert p1.has_text is True
    assert p1.requires_ocr is False
    assert p1.page_type in (PageType.TEXT, PageType.TEXT_AND_IMAGE)


def test_scanned_pdf_inspection():
    pdf_path = FIXTURES_DIR / "scanned_car_policy.pdf"
    assert pdf_path.exists()

    result = inspect_pdf(pdf_path)
    assert result.page_count == 1
    p1 = result.pages[0]
    # Scanned PDF has 0 native text and contains an image
    assert p1.likely_scanned is True
    assert p1.requires_ocr is True
