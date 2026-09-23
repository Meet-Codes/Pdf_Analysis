"""
Unit tests for CandidateResolver:
Verifies priority tiering, single canonical field enforcement, negative validation,
clean make/model separation, and strict JSON serializability.
"""

import json
import pytest
from schemas.base import DocumentType
from retrieval.candidate_resolver import CandidateResolver


def test_priority_tiering_table_over_regex():
    """Test that Priority Tier 4 (Table cell) wins over Priority Tier 2 (Deterministic regex)."""
    resolver = CandidateResolver(DocumentType.MOTOR_INSURANCE)

    # Add regex candidate (Tier 2)
    resolver.add_candidate(
        raw_field="policy_number",
        value="POL-FALLBACK-12345",
        method="deterministic_regex",
        priority_tier=2,
        confidence=0.80
    )

    # Add table cell candidate (Tier 4)
    resolver.add_candidate(
        raw_field="policy_number",
        value="1-12IXB8DM P400",
        method="table_cell",
        priority_tier=4,
        confidence=0.99
    )

    data, evidence = resolver.resolve_all()
    assert data["policy_number"] == "1-12IXB8DM P400"
    assert evidence["policy_number"].method == "table_cell"


def test_single_canonical_field_no_duplicate_aliases():
    """Test that all aliases (CUSTOMER_NAME, insured_name) collapse to exactly ONE customer_name."""
    resolver = CandidateResolver(DocumentType.MOTOR_INSURANCE)

    resolver.add_candidate(raw_field="CUSTOMER_NAME", value="Meet Korat", priority_tier=2)
    resolver.add_candidate(raw_field="insured_name", value="Meet Korat", priority_tier=3)
    resolver.add_candidate(raw_field="customer_name", value="Meet Korat", priority_tier=4)

    data, evidence = resolver.resolve_all()
    assert data["customer_name"] == "Meet Korat"
    # Verify no duplicate alias keys in structured data
    assert "CUSTOMER_NAME" not in data
    assert "insured_name" not in data
    assert "_evidence_candidates" not in data


def test_negative_validation_customer_name():
    """Test that legal boilerplate, brokers, and strings with digits are rejected as customer names."""
    resolver = CandidateResolver(DocumentType.MOTOR_INSURANCE)

    # Boilerplate string
    resolver.add_candidate(raw_field="customer_name", value="DEFENCE OR SETTLEMENT OF PROCEEDINGS", priority_tier=4)
    # Broker string
    resolver.add_candidate(raw_field="customer_name", value="M/s Policy Bazaar Insurance We", priority_tier=3)
    # Valid customer name
    resolver.add_candidate(raw_field="customer_name", value="Bharatbhai M Harsoda", priority_tier=2)

    data, evidence = resolver.resolve_all()
    assert data["customer_name"] == "Bharatbhai M Harsoda"


def test_negative_validation_vehicle_model():
    """Test that 'OF VEHICLE', 'Chassis No.', and column contamination are rejected."""
    resolver = CandidateResolver(DocumentType.MOTOR_INSURANCE)

    resolver.add_candidate(raw_field="vehicle_model", value="OF VEHICLE", priority_tier=4)
    resolver.add_candidate(raw_field="vehicle_model", value="Chassis No.", priority_tier=4)
    resolver.add_candidate(raw_field="vehicle_model", value="ACTIVA/STD", priority_tier=3)

    data, evidence = resolver.resolve_all()
    assert data["vehicle_model"] == "ACTIVA/STD"


def test_make_model_separation():
    """Test that if vehicle_model starts with vehicle_make, make is stripped cleanly."""
    resolver = CandidateResolver(DocumentType.MOTOR_INSURANCE)

    resolver.add_candidate(raw_field="vehicle_make", value="MARUTI", priority_tier=4)
    resolver.add_candidate(raw_field="vehicle_model", value="MARUTI ALTO 800 LXI", priority_tier=3)

    data, evidence = resolver.resolve_all()
    assert data["vehicle_make"] == "MARUTI"
    assert data["vehicle_model"] == "ALTO 800 LXI"


def test_policy_number_checklist_rejection():
    """Test that checklist strings like 'b. Registration Details/RC Copy' are rejected."""
    resolver = CandidateResolver(DocumentType.MOTOR_INSURANCE)

    resolver.add_candidate(raw_field="policy_number", value="b. Registration Details/RC Copy", priority_tier=4)
    resolver.add_candidate(raw_field="policy_number", value="2312101324826900000", priority_tier=3)

    data, evidence = resolver.resolve_all()
    assert data["policy_number"] == "2312101324826900000"


def test_json_serializability():
    """Test that output structured_data contains only pure primitives and dumps cleanly to JSON."""
    resolver = CandidateResolver(DocumentType.MOTOR_INSURANCE)
    raw_data = {
        "customer_name": "Narotambhai Pambhar",
        "policy_number": "D091386640/20012023",
        "registration_number": "GJ03HF4274",
        "vehicle_make": "HONDA",
        "vehicle_model": "ACTIVA/STD",
        "total_premium": "1,942.00",
        "_evidence_candidates": {"dummy": "should_be_stripped"}
    }
    resolver.load_from_raw_extraction(raw_data)
    data, evidence = resolver.resolve_all()

    # Serialization check
    dumped = json.dumps(data)
    assert isinstance(dumped, str)
    assert "_evidence_candidates" not in data
    assert data["total_premium"] == 1942.0
