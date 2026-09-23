"""
Tests for Extraction Prompt Template Complete Connectivity.
Verifies:
1. ExtractionPromptTemplate schema loading (27 fields) and prompt formatting.
2. Robust parsing of LLM responses (markdown-wrapped, raw JSON, regex fallback) and key synchronization.
3. Deterministic motor extraction across all 27 prompt template fields.
4. Normalization of mobile, vehicle class, insurance type, NCB, seating capacity, and financial fields.
5. Canonical visual section building with agent and extended specifications.
6. Grounded Q&A resolution for agent, booking date, vehicle class, mobile, and breakdown.
"""

import pytest
from pathlib import Path
from schemas.base import DocumentType, CanonicalDocument
from agents.prompt_template import (
    extraction_template,
    MOTOR_EXTRACTION_SCHEMA,
    SCHEMA_TO_CANONICAL_MAP,
)
from agents.extractor import extract_deterministic_motor
from validation.normalizer import normalize_fields
from agents.validator import build_canonical_sections
from agents.qa_agent import ask_document


def test_prompt_template_schema_loading_and_formatting():
    """Verify prompt template binds 27 canonical fields and formats system & user prompts."""
    schema = extraction_template.get_schema()
    assert len(schema) == 27, f"Expected 27 schema fields, got {len(schema)}"
    assert "CUSTOMER_NAME" in schema
    assert "TP_PREMIUM" in schema
    assert "AGENT_NAME" in schema
    assert "NCB" in schema
    assert "CNG_IDV" in schema

    # Check system prompt includes schema guidelines
    sys_prompt = extraction_template.get_system_prompt()
    assert "You are a document extraction engine" in sys_prompt
    assert "CUSTOMER_NAME" in sys_prompt
    assert "TOTAL_PREMIUM" in sys_prompt

    # Check user prompt formatting
    user_prompt = extraction_template.format_user_prompt(
        text="Sample policy text content for MH12AB1234",
        doc_type=DocumentType.MOTOR_INSURANCE,
        target_fields=["CUSTOMER_NAME", "POLICY_NUMBER", "AGENT_CODE"]
    )
    assert "Sample policy text content" in user_prompt
    assert "CUSTOMER_NAME" in user_prompt
    assert "AGENT_CODE" in user_prompt


def test_prompt_template_llm_response_parsing():
    """Verify robust parsing of markdown-fenced LLM output and bidirectional key synchronization."""
    raw_llm_json = """
    ```json
    {
        "CUSTOMER_NAME": "Mr Meet Korat",
        "CUSTOMER_MOBILE": "9876543210",
        "COMPANY_NAME": "TATA AIG GENERAL INSURANCE COMPANY LIMITED",
        "AGENT_NAME": "Rajesh Sharma",
        "AGENT_CODE": "AGT9988",
        "CLASS_OF_VEHICLE": "commercial",
        "INSURANCE_TYPE": "comprehensive/package",
        "POLICY_BOOKING_DATE": "10/01/2024",
        "POLICY_START_DATE": "15/01/2024",
        "POLICY_END_DATE": "14/01/2025",
        "POLICY_NUMBER": "TATA/CV/2024/991122",
        "VEHICLE_REGISTRATION_NUMBER": "GJ03MG6586",
        "TP_PREMIUM": "12500.00",
        "OD_PREMIUM": "3500.00",
        "NET_PREMIUM": "16000.00",
        "ADDON_PREMIUM": "1200.00",
        "GST_AMOUNT": "3096.00",
        "TOTAL_PREMIUM": "20296.00",
        "TOTAL_IDV": "450000.00",
        "CNG_IDV": "30000.00",
        "ENGINE_NUMBER": "ENG998811",
        "CHASSIS_NUMBER": "CHA776655",
        "YEAR_OF_MANUFACTURE": "2023",
        "MAKE": "TATA",
        "MODEL": "ACE 275",
        "SEATING_CAPACITY": "2",
        "NCB": "20%"
    }
    ```
    """
    parsed = extraction_template.parse_llm_response(raw_llm_json)
    assert parsed["CUSTOMER_NAME"] == "Mr Meet Korat"
    # Canonical lowercase alias synchronization
    assert parsed["customer_name"] == "Mr Meet Korat"
    assert parsed["insured_name"] == "Mr Meet Korat"
    assert parsed["agent_name"] == "Rajesh Sharma"
    assert parsed["agent_code"] == "AGT9988"
    assert parsed["policy_number"] == "TATA/CV/2024/991122"
    assert parsed["registration_number"] == "GJ03MG6586"
    assert parsed["total_premium"] == "20296.00"
    assert parsed["cng_idv"] == "30000.00"


def test_deterministic_motor_prompt_schema_fields():
    """Verify deterministic extractor captures newly specified schema fields from raw text."""
    sample_text = """
    Reliance General Insurance Company Limited
    Policy Number: REL-MOT-2024-554433
    Name of Insured: Mr. Vikram Singh
    Mobile No: 9876543210
    Intermediary Name: Rahul Mehta
    Intermediary Code: AGT-5544
    Class of Vehicle: Commercial Goods Vehicle
    Coverage: Comprehensive Package Policy
    Booking Date: 05/02/2024
    Period of Insurance From: 10/02/2024 To: 09/02/2025
    Registration No: RJ 14 GA 8899
    Make: TATA Model: 407 TRUCK
    Engine No: ENG77665544
    Chassis No: CHA99887766
    Manufacturing Year: 2022
    Seating Capacity: 3
    Vehicle IDV: 5,50,000.00
    CNG Kit IDV: 25,000.00
    NCB: 25%
    Basic OD: 6,500.00
    Total TP: 10,000.00
    Net Premium: 16,500.00
    Add-on Premium: 1,500.00
    GST: 3,240.00
    Total Amount Payable: 21,240.00
    """
    data = extract_deterministic_motor(sample_text)
    assert data["policy_number"] == "REL-MOT-2024-554433"
    assert data["insured_name"] == "Mr. Vikram Singh"
    assert data["customer_mobile"] == "9876543210"
    assert data["agent_name"] == "Rahul Mehta"
    assert data["agent_code"] == "AGT-5544"
    assert data["class_of_vehicle"] == "commercial"
    assert data["insurance_type"] == "comprehensive/package"
    assert data["policy_booking_date"] == "05/02/2024"
    assert data["registration_number"] == "RJ 14 GA 8899"
    assert data["year_of_manufacture"] == "2022"
    assert data["seating_capacity"] == "3"
    assert data["total_idv"] == "5,50,000.00"
    assert data["cng_idv"] == "25,000.00"
    assert data["ncb"] == "25%"
    assert data["net_premium"] == "16,500.00"
    assert data["addon_premium"] == "1,500.00"
    assert data["total_premium"] == "21,240.00"


def test_normalizer_prompt_schema_fields():
    """Verify normalizer cleans all 27 prompt schema fields into standardized formats."""
    raw = {
        "customer_name": "MR. MEET KORAT",
        "customer_mobile": "9876543210",
        "agent_name": "mr rajesh sharma",
        "agent_code": "agt - 1234",
        "class_of_vehicle": "Private Car (Saloon)",
        "insurance_type": "Comprehensive Package Policy",
        "policy_booking_date": "01-05-2024",
        "policy_number": "POL / 9988 / 001",
        "vehicle_registration_number": "GJ 03 MG 6586",
        "net_premium": "15,430.00",
        "addon_premium": "1,200.00",
        "gst_amount": "2,777.00",
        "total_premium": "19,407.00",
        "total_idv": "4,50,000.00",
        "cng_idv": "30,000.00",
        "ncb": "20",
        "seating_capacity": "5 seats",
        "year_of_manufacture": "Mfg 2023",
    }
    norm = normalize_fields(raw, DocumentType.MOTOR_INSURANCE)
    assert norm["customer_name"] == "Mr Meet Korat"
    assert norm["customer_mobile"] == "+91 98765 43210"
    assert norm["agent_name"] == "Mr Rajesh Sharma"
    assert norm["class_of_vehicle"] == "private car"
    assert norm["insurance_type"] == "comprehensive/package"
    assert norm["policy_booking_date"] == "1 May 2024"
    assert norm["vehicle_registration_number"] == "GJ03MG6586"
    assert norm["net_premium"] == 15430.0
    assert norm["addon_premium"] == 1200.0
    assert norm["gst_amount"] == 2777.0
    assert norm["total_premium"] == 19407.0
    assert norm["total_idv"] == 450000.0
    assert norm["cng_idv"] == 30000.0
    assert norm["ncb"] == "20%"
    assert norm["seating_capacity"] == "5"
    assert norm["year_of_manufacture"] == "2023"


def test_canonical_sections_prompt_schema_fields():
    """Verify visual presentation sections display the extended prompt schema fields."""
    data = {
        "insured_name": "Mr Meet Korat",
        "mobile": "+91 98765 43210",
        "policy_number": "P0023200023/4115/103739",
        "insurance_type": "comprehensive/package",
        "company_name": "TATA AIG General Insurance",
        "agent_name": "Mr Rajesh Sharma",
        "agent_code": "AGT-101",
        "registration_number": "GJ03MG6586",
        "class_of_vehicle": "commercial",
        "vehicle_make": "TATA",
        "vehicle_model": "ACE 275 ID",
        "manufacturing_year": "2023",
        "seating_capacity": "2",
        "total_idv": 450000.0,
        "cng_idv": 30000.0,
        "own_damage_premium": 3500.0,
        "third_party_premium": 12500.0,
        "net_premium": 16000.0,
        "addon_premium": 1200.0,
        "gst": 3096.0,
        "ncb": "20%",
        "total_premium": 20296.0,
        "policy_booking_date": "10 January 2024",
        "policy_start_date": "15 January 2024",
        "policy_end_date": "14 January 2025",
    }
    sections = build_canonical_sections(data, DocumentType.MOTOR_INSURANCE)
    labels = [f["label"] for s in sections for f in s["fields"]]
    assert "Agent Name" in labels
    assert "Agent Code" in labels
    assert "Class of Vehicle" in labels
    assert "Vehicle IDV" in labels
    assert "CNG/LPG IDV" in labels
    assert "Add-on Premium" in labels
    assert "Net Premium" in labels
    assert "Booking Date" in labels
    assert "NCB" in labels


def test_grounded_qa_prompt_schema_fields():
    """Verify Q&A directly resolves questions about agent, vehicle class, booking date, NCB, CNG IDV."""
    canonical = CanonicalDocument(
        document_id="test_qa_prompt_doc",
        file_name="test_policy.pdf",
        document_type=DocumentType.MOTOR_INSURANCE,
        title="Motor Insurance Policy",
        summary="Test summary",
        structured_data={
            "agent_name": "Mr Rajesh Sharma",
            "agent_code": "AGT-9988",
            "class_of_vehicle": "commercial",
            "insurance_type": "comprehensive/package",
            "policy_booking_date": "10 January 2024",
            "customer_mobile": "+91 98765 43210",
            "addon_premium": 1200.0,
            "cng_idv": 30000.0,
            "seating_capacity": "2",
            "ncb": "20%",
        },
        sections=[],
        evidence={},
        is_valid=True,
    )

    ans_agent = ask_document("Who is the agent?", canonical)
    assert "Rajesh Sharma" in ans_agent

    ans_code = ask_document("What is the agent code?", canonical)
    assert "AGT-9988" in ans_code

    ans_class = ask_document("What class is the vehicle?", canonical)
    assert "commercial" in ans_class

    ans_type = ask_document("What is the insurance type?", canonical)
    assert "comprehensive/package" in ans_type

    ans_book = ask_document("When was the policy booked?", canonical)
    assert "10 January 2024" in ans_book

    ans_ncb = ask_document("What is the NCB?", canonical)
    assert "20%" in ans_ncb

    ans_seats = ask_document("What is the seating capacity?", canonical)
    assert "2" in ans_seats
