"""
Unit tests for Field Normalization.
Section 60: Normalization Acceptance Test suite.
"""

import pytest
from validation.field_rules import (
    clean_customer_name,
    clean_registration_number,
    clean_policy_number,
    clean_phone_number,
)
from validation.dates import parse_and_normalize_date
from validation.financial import parse_currency_amount
from utils.formatting import format_currency_inr


def test_customer_name_normalization():
    # Acceptance Test 1
    assert clean_customer_name("MR. MEET KORAT") == "Mr Meet Korat"

    # Acceptance Test 2 (with slash and ID)
    assert clean_customer_name("MR MEET KORAT / 3654789/5") == "Mr Meet Korat"

    # Acceptance Test 3 (with enterprise prefix)
    assert clean_customer_name("MHL ABC MR MEET KORAT 3654789/5") == "Mr Meet Korat"


def test_registration_number_normalization():
    # Acceptance Test
    assert clean_registration_number("GJ 03 MG 6586") == "GJ03MG6586"
    assert clean_registration_number("DL 1C AA 1111") == "DL1CAA1111"


def test_policy_number_normalization():
    # Acceptance Test
    assert clean_policy_number("P0023200023 / 4115 / 103739") == "P0023200023/4115/103739"


def test_financial_normalization():
    # Acceptance Test
    amt = parse_currency_amount("Rs. 1,25,000/-")
    assert amt == 125000.0
    assert format_currency_inr(amt) == "₹1,25,000"

    amt2 = parse_currency_amount("Rs. 15,430/-")
    assert amt2 == 15430.0
    assert format_currency_inr(amt2) == "₹15,430"


def test_date_normalization():
    # Acceptance Test
    assert parse_and_normalize_date("29/03/2023") == "29 March 2023"
    assert parse_and_normalize_date("2023-03-29") == "29 March 2023"
