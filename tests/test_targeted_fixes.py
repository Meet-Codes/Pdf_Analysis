"""
Targeted Verification Tests for Strict Non-Regression Fixes:
1. Rejection of invalid customer_name values (Customer Id, M, Policy Number, Chassis, etc.)
2. Rejection of invalid policy_number values (Plan names like Support Plus, generic text without digits)
3. Rejection of invalid engine_number and chassis_number values (Chassis, Engine No., etc.)
4. Financial field associations (TOTAL not mapped to tax, CGST/SGST not mapped to total_premium, tax < total_premium rule)
5. Liability-only policies keep IDV and OD premium as None
6. Domain field preservation for PROPERTY_INSURANCE and WORKMEN_COMPENSATION
7. Canonical data clean serializability (JSON primitives, no candidate objects)
"""

import json
import pytest
from schemas.base import DocumentType
from validation.field_rules import (
    clean_customer_name,
    clean_policy_number,
    clean_engine_number,
    clean_chassis_number,
)
from validation.financial import parse_currency_amount
from retrieval.candidate_resolver import CandidateResolver
from retrieval.field_resolvers.amount_resolver import AmountResolver
from retrieval.field_resolvers.customer_name_resolver import CustomerNameResolver
from ui.document_view import clean_canonical_data
from schemas.property import PropertyInsuranceSchema
from schemas.workmen import WorkmenCompensationSchema


def test_customer_name_validation():
    # Invalid names must be rejected
    assert clean_customer_name("Customer Id") is None
    assert clean_customer_name("Customer ID") is None
    assert clean_customer_name("M") is None
    assert clean_customer_name("A") is None
    assert clean_customer_name("Policy Number") is None
    assert clean_customer_name("Chassis") is None
    assert clean_customer_name("Engine No.") is None
    assert clean_customer_name("Registration No") is None
    assert clean_customer_name("Plan Name") is None

    # Valid names must be accepted and properly title-cased
    assert clean_customer_name("RAHUL SHARMA") == "Rahul Sharma"
    assert clean_customer_name("Tata Motors Ltd") == "Tata Motors Ltd"
    assert clean_customer_name("Meet Korat") == "Meet Korat"


def test_policy_number_validation():
    # Invalid policy numbers (plan names, labels, text without digits)
    assert clean_policy_number("Support Plus") is None
    assert clean_policy_number("Motor Secure Plan") is None
    assert clean_policy_number("Policy Number") is None
    assert clean_policy_number("Address") is None
    assert clean_policy_number("Comprehensive Package Policy") is None

    # Valid policy numbers with digits
    assert clean_policy_number("OG-20-1234-1801-000001") == "OG-20-1234-1801-000001"
    assert clean_policy_number("1-12IXB8DM P400") == "1-12IXB8DM P400"
    assert clean_policy_number("1234/56789/00/000") == "1234/56789/00/000"


def test_engine_and_chassis_number_validation():
    # Engine number should reject label collisions
    assert clean_engine_number("Chassis") is None
    assert clean_engine_number("CHASSIS") is None
    assert clean_engine_number("Engine No.") is None
    assert clean_engine_number("Registration No") is None
    assert clean_engine_number("1NZFE123456") == "1NZFE123456"

    # Chassis number should reject label collisions
    assert clean_chassis_number("Engine") is None
    assert clean_chassis_number("ENGINE NO") is None
    assert clean_chassis_number("Chassis No.") is None
    assert clean_chassis_number("MA3EWFB1S00123456") == "MA3EWFB1S00123456"


def test_financial_field_association():
    resolver = AmountResolver()

    # 1. Standalone "TOTAL 4090" must NOT map to tax
    text = "Total Tax / GST: 736.20\nTOTAL AMOUNT: 4090.00"
    resolved = resolver.resolve(full_text=text, page_texts={1: text})

    assert resolved.get("total_premium") is not None
    assert parse_currency_amount(str(resolved["total_premium"].value)) == 4090.00
    assert resolved.get("tax") is not None
    assert parse_currency_amount(str(resolved["tax"].value)) == 736.20
    # Tax must NOT be 4090.00
    assert parse_currency_amount(str(resolved["tax"].value)) != 4090.00


def test_financial_tax_less_than_total_rule():
    resolver = AmountResolver()

    # If an erroneous tax candidate is greater than or equal to total_premium, it must be discarded
    text = "Net Premium: 3354.00\nTotal Amount: 4090.00\nTax: 4090.00"
    resolved = resolver.resolve(full_text=text, page_texts={1: text})
    assert resolved.get("total_premium") is not None
    # Tax candidate was >= total_premium, so _cross_validate_financials discarded it
    assert resolved.get("tax") is None


def test_liability_only_policy_idv_and_od_are_none():
    resolver = AmountResolver()

    text = "MOTOR LIABILITY ONLY POLICY CERTIFICATE OF INSURANCE\nBasic TP Premium: 2094.00\nTotal Premium: 2471.00"
    resolved = resolver.resolve(full_text=text, page_texts={1: text})

    # For liability-only, IDV and basic OD premium must be None
    assert resolved.get("idv") is None
    assert resolved.get("basic_od_premium") is None
    assert resolved.get("basic_tp_premium") is not None
    assert parse_currency_amount(str(resolved["basic_tp_premium"].value)) == 2094.00
    assert resolved.get("total_premium") is not None
    assert parse_currency_amount(str(resolved["total_premium"].value)) == 2471.00


def test_property_insurance_fields_preserved():
    resolver = CandidateResolver(DocumentType.PROPERTY_INSURANCE)

    resolver.add_candidate("insured_business", "Acme Industries Pvt Ltd", priority_tier=3)
    resolver.add_candidate("policy_number", "PROP-2024-9988", priority_tier=3)
    resolver.add_candidate("total_sum_insured", 5000000.0, priority_tier=3)
    resolver.add_candidate("base_premium", 12500.0, priority_tier=3)
    resolver.add_candidate("total_premium", 14750.0, priority_tier=3)

    data, evidence = resolver.resolve_all()

    # Domain specific fields must NOT be discarded
    assert data["insured_business"] == "Acme Industries Pvt Ltd"
    assert data["policy_number"] == "PROP-2024-9988"
    assert data["total_sum_insured"] == 5000000.0
    assert data["base_premium"] == 12500.0
    assert data["total_premium"] == 14750.0


def test_workmen_compensation_fields_preserved():
    resolver = CandidateResolver(DocumentType.WORKMEN_COMPENSATION)

    resolver.add_candidate("employer", "Buildcon Infrastructure Ltd", priority_tier=3)
    resolver.add_candidate("policy_number", "WC-2024-1122", priority_tier=3)
    resolver.add_candidate("number_of_employees", 45, priority_tier=3)
    resolver.add_candidate("sum_insured", 3600000.0, priority_tier=3)
    resolver.add_candidate("total_premium", 42480.0, priority_tier=3)

    data, evidence = resolver.resolve_all()

    # Domain specific fields must NOT be discarded
    assert data["employer"] == "Buildcon Infrastructure Ltd"
    assert data["policy_number"] == "WC-2024-1122"
    assert data["number_of_employees"] == 45
    assert data["sum_insured"] == 3600000.0
    assert data["total_premium"] == 42480.0


def test_clean_canonical_data_serializability():
    # Test that clean_canonical_data produces pure JSON serializable dict without candidate objects
    resolver = CandidateResolver(DocumentType.MOTOR_INSURANCE)
    resolver.add_candidate("customer_name", "Amit Verma", priority_tier=3)
    resolver.add_candidate("policy_number", "POL-987654", priority_tier=3)
    resolver.add_candidate("total_premium", 5500.0, priority_tier=3)

    data, evidence = resolver.resolve_all()

    # Wrap in nested structure with candidate-like keys
    raw_structure = {
        "canonical": data,
        "_internal_meta": "hidden",
        "candidate_list": [1, 2, 3],
        "nested": {
            "evidence": evidence.get("customer_name"),
            "amount": 5500.0
        }
    }

    clean = clean_canonical_data(raw_structure)

    # Must be 100% JSON dumpable
    serialized = json.dumps(clean)
    deserialized = json.loads(serialized)

    assert "_internal_meta" not in deserialized
    assert "candidate_list" not in deserialized
    assert deserialized["nested"]["evidence"] == "Amit Verma"
    assert deserialized["canonical"]["customer_name"] == "Amit Verma"


def test_forbidden_customer_name_tokens_rejected():
    forbidden = [
        "Person", "Customer", "Customer Id", "Customer ID", "Policyholder",
        "Policyholder Name", "Name", "Name of Insured Person", "Insured Person",
        "Insured Person(s)", "Self", "Spouse", "Daughter", "Son", "Father",
        "Mother", "Relationship", "Date of Birth", "Gender", "Member ID"
    ]
    for token in forbidden:
        assert clean_customer_name(token) is None, f"Failed: {token} was not rejected"


def test_health_insurance_policyholder_name_extracted_over_insured_members():
    """
    Test 1 & 2: Health document with Policyholder Name where policyholder
    and insured members are different, and Insured Person(s) table contains 'Person'.
    Must extract policyholder name, reject 'Person', and not confuse with members.
    """
    text = (
        "STAR HEALTH AND ALLIED INSURANCE CO LTD\n"
        "Policyholder Name\n"
        "Mr DEVABHAI NARSHIBHAI TOPIYA\n"
        "Insured Person(s) Details:\n"
        "Name of Insured Person | Relationship | Age\n"
        "Person | Self | 42\n"
        "Rina Devabhai Topiya | Spouse | 38\n"
        "Prisha Devabhai Topiya | Daughter | 12\n"
    )
    resolver = CustomerNameResolver()
    cand = resolver.resolve(full_text=text, page_texts={1: text}, context={"doc_type": DocumentType.HEALTH_INSURANCE})
    assert cand is not None
    # Must be Mr Devabhai Narshibhai Topiya
    assert cand.value == "Mr Devabhai Narshibhai Topiya"
    assert cand.value != "Person"
    assert "Topiya" in cand.value

    # In CandidateResolver:
    c_res = CandidateResolver(DocumentType.HEALTH_INSURANCE)
    c_res.add_candidate("customer_name", cand.value, priority_tier=3)
    c_res.add_candidate("policy_number", "P/123456/01/2024/001234", priority_tier=3)
    structured, _ = c_res.resolve_all()
    assert structured["customer_name"] == "Mr Devabhai Narshibhai Topiya"
    assert structured["policyholder_name"] == "Mr Devabhai Narshibhai Topiya"
    assert structured["customer_name"] != "Person"


def test_health_insurance_without_policyholder_name_returns_none():
    """
    Test 3: Health document without Policyholder Name label.
    Must return None, NEVER substitute 'Person', 'Insured Person', or a random member.
    """
    text = (
        "HEALTH INSURANCE POLICY SCHEDULE\n"
        "Insured Person(s) Details:\n"
        "Name of Insured Person | Relationship | Member ID\n"
        "Person | Self | MEM001\n"
        "Sum Insured: 5,00,000\n"
    )
    resolver = CustomerNameResolver()
    cand = resolver.resolve(full_text=text, page_texts={1: text}, context={"doc_type": DocumentType.HEALTH_INSURANCE})
    # Cannot accept "Person" or table header tokens
    assert cand is None


def test_motor_insurance_customer_name_non_regression():
    """
    Test 4: Motor insurance customer name extraction remains completely unchanged.
    """
    text = (
        "MOTOR VEHICLE INSURANCE CERTIFICATE\n"
        "Name of Insured: Shri Rajesh Kumar Gupta\n"
        "Address: 123 MG Road, Rajkot\n"
        "Policy No: 1234/5678/00\n"
    )
    resolver = CustomerNameResolver()
    cand = resolver.resolve(full_text=text, page_texts={1: text}, context={"doc_type": DocumentType.MOTOR_INSURANCE})
    assert cand is not None
    assert cand.value == "Shri Rajesh Kumar Gupta"


def test_property_insurance_customer_name_non_regression():
    """
    Test 5: Property insurance customer name / insured business remains completely unchanged.
    """
    text = (
        "STANDARD FIRE AND SPECIAL PERILS POLICY\n"
        "Insured Name: Acme Industries Private Limited\n"
        "Policy Number: PROP-2024-9988\n"
    )
    resolver = CustomerNameResolver()
    cand = resolver.resolve(full_text=text, page_texts={1: text}, context={"doc_type": DocumentType.PROPERTY_INSURANCE})
    assert cand is not None
    assert "Acme Industries" in cand.value

