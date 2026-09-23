"""
Unit and Regression Tests for Insurance Financial & Premium Extraction.
Verifies all 8 canonical financial fields:
- idv, cng_idv, ncb, basic_od_premium, basic_tp_premium, net_premium, tax, total_premium.
Tests Indian monetary formats, multi-column tables, stacked layouts, arithmetic reconciliation,
and strict negative validation (rejecting dates, phone numbers, policy numbers, GSTINs).
"""

import pytest
from retrieval.field_resolvers.amount_resolver import AmountResolver
from retrieval.candidate_resolver import CandidateResolver
from validation.financial import parse_currency_amount
from validation.field_rules import clean_ncb
from schemas.base import DocumentType
from ingestion.document_layout import DocumentLayoutModel, LineBox, TokenBox


def test_indian_currency_formats():
    """Requirement 5 & 15: Tests Indian monetary formats and currency symbols."""
    samples = [
        ("₹12,345.00", 12345.0),
        ("Rs. 12,345", 12345.0),
        ("INR 12,345.50", 12345.5),
        ("1,23,456.00", 123456.0),
        ("1,23,456", 123456.0),
        ("₹ 12,345", 12345.0),
        ("12345.00", 12345.0),
        ("Rs 12345", 12345.0),
    ]
    for raw_str, expected in samples:
        val = parse_currency_amount(raw_str)
        assert val == expected, f"Failed parsing {raw_str}: got {val}, expected {expected}"


def test_amount_same_line_as_label():
    """Requirement 8 & 15: Tests amounts appearing on the same line as label."""
    text = (
        "Basic OD Premium: ₹ 4,532.00\n"
        "Basic TP Premium: ₹ 1,432.00\n"
        "Net Premium: ₹ 5,964.00\n"
        "GST: ₹ 1,073.52\n"
        "Total Premium: ₹ 7,037.52\n"
    )
    ar = AmountResolver()
    res = ar.resolve(full_text=text, page_texts={1: text})

    assert res["basic_od_premium"] is not None
    assert parse_currency_amount(res["basic_od_premium"].value) == 4532.0

    assert res["basic_tp_premium"] is not None
    assert parse_currency_amount(res["basic_tp_premium"].value) == 1432.0

    assert res["net_premium"] is not None
    assert parse_currency_amount(res["net_premium"].value) == 5964.0

    assert res["tax"] is not None
    assert parse_currency_amount(res["tax"].value) == 1073.52

    assert res["total_premium"] is not None
    assert parse_currency_amount(res["total_premium"].value) == 7037.52


def test_amount_in_separate_table_column():
    """Requirement 4 & 15: Tests amounts appearing in a separate table column."""
    tables = [{
        "page_number": 1,
        "table_index": 0,
        "rows": [
            ["Coverage", "IDV in Rs."],
            ["Vehicle IDV", "450000.00"],
            ["Bi-Fuel Kit", "25000.00"],
            ["Total Value", "475000.00"],
        ]
    }]
    ar = AmountResolver()
    res = ar.resolve(full_text="", page_texts={1: ""}, tables=tables)

    assert res["idv"] is not None
    assert parse_currency_amount(res["idv"].value) == 450000.0

    assert res["cng_idv"] is not None
    assert parse_currency_amount(res["cng_idv"].value) == 25000.0


def test_multiple_amounts_in_same_row():
    """Requirement 8 & 15: Tests multiple amounts in the same table row."""
    tables = [{
        "page_number": 1,
        "table_index": 0,
        "rows": [
            ["A. Own Damage Premium(Rs.)", "", "", "B. Third Party Premium(Rs.)", ""],
            ["Basic Premium", "3989.00", "", "Basic Premium", "1850.00"],
            ["Total Net Premium", "5839.00", "", "Total Invoice Value", "6889.00"],
        ]
    }]
    ar = AmountResolver()
    res = ar.resolve(full_text="", page_texts={1: ""}, tables=tables)

    assert res["basic_od_premium"] is not None
    assert parse_currency_amount(res["basic_od_premium"].value) == 3989.0

    assert res["basic_tp_premium"] is not None
    assert parse_currency_amount(res["basic_tp_premium"].value) == 1850.0


def test_vertical_column_aligned_layout():
    """Requirement 4 & 8: Tests vertical column-aligned layout (header above, value below)."""
    layout = DocumentLayoutModel(document_id="layout_test")

    # Header line
    header_line = LineBox(
        text="Net Premium    CGST    SGST    Gross Premium",
        bbox=(170.0, 390.0, 550.0, 402.0),
        page=1,
        line_no=1,
        tokens=[
            TokenBox("Net", (174.0, 390.0, 190.0, 402.0), page=1),
            TokenBox("Premium", (192.0, 390.0, 225.0, 402.0), page=1),
            TokenBox("CGST", (305.0, 390.0, 330.0, 402.0), page=1),
            TokenBox("SGST", (355.0, 390.0, 380.0, 402.0), page=1),
            TokenBox("Gross", (490.0, 390.0, 515.0, 402.0), page=1),
            TokenBox("Premium", (518.0, 390.0, 550.0, 402.0), page=1),
        ]
    )

    # Value line
    value_line = LineBox(
        text="714.00    64.26    64.26    842.52",
        bbox=(205.0, 407.0, 550.0, 418.0),
        page=1,
        line_no=2,
        tokens=[
            TokenBox("714.00", (208.0, 407.0, 235.0, 418.0), page=1),
            TokenBox("64.26", (310.0, 407.0, 335.0, 418.0), page=1),
            TokenBox("64.26", (360.0, 407.0, 385.0, 418.0), page=1),
            TokenBox("842.52", (520.0, 407.0, 548.0, 418.0), page=1),
        ]
    )

    all_tokens = header_line.tokens + value_line.tokens
    layout.add_page_elements(page=1, tokens=all_tokens, lines=[header_line, value_line], blocks=[])

    ar = AmountResolver()
    res = ar.resolve(full_text="", page_texts={1: ""}, layout=layout)

    assert res["net_premium"] is not None
    assert parse_currency_amount(res["net_premium"].value) == 714.0

    assert res["total_premium"] is not None
    assert parse_currency_amount(res["total_premium"].value) == 842.52

    assert res["tax"] is not None
    assert parse_currency_amount(res["tax"].value) == 128.52  # Dual tax 64.26 * 2 reconciled!


def test_ncb_percentage_and_amount_formats():
    """Requirement 7 & 15: Tests NCB as percentage or discount amount."""
    assert clean_ncb("20%") == "20%"
    assert clean_ncb("20") == "20%"
    assert clean_ncb("0.0%") == "0.0%"
    assert clean_ncb("0%") == "0%"
    assert clean_ncb("50%") == "50%"
    assert clean_ncb("Rs. 1,755.50") == "1755.5"
    assert clean_ncb("500.00") == "500.0"


def test_rejection_of_unrelated_numbers():
    """Requirement 6: Rejects dates, phone numbers, policy numbers, and GSTINs from premium fields."""
    ar = AmountResolver()
    # Dates
    assert not ar._is_valid_monetary("10/01/2023", "total_premium")
    assert not ar._is_valid_monetary("2023-01-20", "net_premium")
    # Phone number
    assert not ar._is_valid_monetary("9998621574", "total_premium")
    # PIN code
    assert not ar._is_valid_monetary("360021", "basic_od_premium")
    # GSTIN
    assert not ar._is_valid_monetary("27AABCL5045N1Z8", "tax")
    # Policy number
    assert not ar._is_valid_monetary("2312101324826900000", "total_premium")
    # Zero for total premium
    assert not ar._is_valid_monetary("0.00", "total_premium")


def test_fail_closed_null_behavior():
    """Requirement 14: If a field genuinely does not exist in the document, return None."""
    resolver = CandidateResolver(doc_type=DocumentType.MOTOR_INSURANCE)
    # Only load basic info without CNG or NCB
    resolver.add_candidate(raw_field="policy_number", value="POL-999")
    resolver.add_candidate(raw_field="total_premium", value="5000.0")

    canonical, _ = resolver.resolve_all()
    assert canonical["idv"] is None
    assert canonical["cng_idv"] is None
    assert canonical["ncb"] is None
    assert canonical["basic_od_premium"] is None
    assert canonical["basic_tp_premium"] is None
    assert canonical["total_premium"] == 5000.0
