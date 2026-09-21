"""
Unit tests for Document Classifier.
Rule 6 & 7: Document classification must be based on actual document content, NEVER filename.
"""

from pathlib import Path
import pytest
from ingestion.text_extractor import extract_native_text
from agents.classifier import classify_document_content
from schemas.base import DocumentType

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_misleading_filename_classification():
    """
    CRITICAL ACCEPTANCE TEST:
    A file named 'PANCARD_17041098095175.pdf' containing electricity bill text
    MUST be classified as ELECTRICITY_BILL, never IDENTITY_DOCUMENT.
    """
    pdf_path = FIXTURES_DIR / "PANCARD_17041098095175.pdf"
    assert pdf_path.exists()

    text_dict = extract_native_text(pdf_path)
    full_text = " ".join([d["text"] for d in text_dict.values()])

    doc_type, subtype = classify_document_content(full_text)
    assert doc_type == DocumentType.ELECTRICITY_BILL


def test_motor_insurance_classification():
    pdf_path = FIXTURES_DIR / "tata_ace_insurance.pdf"
    assert pdf_path.exists()

    text_dict = extract_native_text(pdf_path)
    full_text = " ".join([d["text"] for d in text_dict.values()])

    doc_type, subtype = classify_document_content(full_text)
    assert doc_type == DocumentType.MOTOR_INSURANCE
    assert subtype in ("COMMERCIAL_VEHICLE", "CAR")


def test_health_insurance_classification():
    pdf_path = FIXTURES_DIR / "health_insurance.pdf"
    assert pdf_path.exists()

    text_dict = extract_native_text(pdf_path)
    full_text = " ".join([d["text"] for d in text_dict.values()])

    doc_type, subtype = classify_document_content(full_text)
    assert doc_type == DocumentType.HEALTH_INSURANCE
