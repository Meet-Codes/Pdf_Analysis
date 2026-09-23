"""
End-to-End Automated Pipeline Test.
Sections 58, 59: Runs complete pipeline on sample test documents and verifies
structured extraction, normalization, validation, canonical output, and grounded Q&A.
"""

from pathlib import Path
import pytest
from schemas.base import DocumentType
from agents.workflow import process_document
from agents.qa_agent import ask_document

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_end_to_end_motor_insurance():
    pdf_path = FIXTURES_DIR / "tata_ace_insurance.pdf"
    assert pdf_path.exists()

    doc_id = "test_doc_motor_e2e"
    canonical = process_document(pdf_path, doc_id, "tata_ace_insurance.pdf")

    # 1. Verify classification
    assert canonical.document_type == DocumentType.MOTOR_INSURANCE

    # 2. Verify normalized values
    data = canonical.structured_data
    assert (data.get("customer_name") or data.get("insured_name")) == "Mr Meet Korat"
    assert data.get("policy_number") == "P0023200023/4115/103739"
    assert data.get("registration_number") == "GJ03MG6586"
    assert data.get("vehicle_make") == "TATA"
    assert data.get("policy_start_date") == "29 March 2023"
    assert data.get("policy_end_date") == "28 March 2024"
    assert data.get("total_premium") == 18207.0

    # 3. Verify validation
    assert canonical.is_valid is True

    # 4. Verify summary content
    assert "Mr Meet Korat" in canonical.summary
    assert "P0023200023/4115/103739" in canonical.summary
    assert "GJ03MG6586" in canonical.summary

    # 5. Verify Grounded Q&A
    ans_name = ask_document("What is the customer's name?", canonical)
    assert "Mr Meet Korat" in ans_name

    ans_pol = ask_document("What is the policy number?", canonical)
    assert "P0023200023/4115/103739" in ans_pol

    ans_reg = ask_document("What is the registration number?", canonical)
    assert "GJ03MG6586" in ans_reg

    ans_exp = ask_document("When does this policy expire?", canonical)
    assert "28 March 2024" in ans_exp

    ans_prem = ask_document("What is the total premium?", canonical)
    assert "18,207" in ans_prem

    # 6. Verify Fail-Closed policy on absent information
    ans_pan = ask_document("What is the customer's PAN card number?", canonical)
    assert ans_pan == "The document does not contain this information."


def test_end_to_end_misleading_filename_electricity():
    """
    Acceptance Test: Misleading filename 'PANCARD_17041098095175.pdf'
    must process end-to-end as ELECTRICITY_BILL.
    """
    pdf_path = FIXTURES_DIR / "PANCARD_17041098095175.pdf"
    assert pdf_path.exists()

    doc_id = "test_doc_elec_e2e"
    canonical = process_document(pdf_path, doc_id, "PANCARD_17041098095175.pdf")

    assert canonical.document_type == DocumentType.ELECTRICITY_BILL
    assert canonical.structured_data.get("consumer_number") == "12345678901"
    assert canonical.structured_data.get("customer_name") == "Mr Meet Korat"
    assert canonical.structured_data.get("total_bill") == 2500.0
