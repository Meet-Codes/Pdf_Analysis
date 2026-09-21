"""
Unit tests for Validator Engine and Cross-field checks.
"""

import pytest
from validation.validator import validate_document_data
from validation.dates import validate_date_range
from validation.financial import validate_premium_components
from schemas.base import DocumentType


def test_date_range_validation():
    # Valid
    valid, msg = validate_date_range("29 March 2023", "28 March 2024")
    assert valid is True
    assert msg is None

    # Invalid: start after end
    valid_inv, msg_inv = validate_date_range("28 March 2024", "29 March 2023")
    assert valid_inv is False
    assert "cannot be later than" in msg_inv


def test_premium_arithmetic_reconciliation():
    # 15430 + 2777 = 18207 -> matches!
    valid, msg = validate_premium_components(
        own_damage=0.0,
        third_party=15430.0,
        gst=2777.0,
        total=18207.0,
    )
    assert valid is True

    # Bad math: 1000 + 100 != 5000
    valid_bad, msg_bad = validate_premium_components(
        own_damage=1000.0,
        third_party=0.0,
        gst=100.0,
        total=5000.0,
    )
    assert valid_bad is False
    assert "differs from declared total" in msg_bad


def test_cross_field_collision():
    # If registration equals engine number, flag error
    data = {
        "registration_number": "GJ03MG6586",
        "engine_number": "GJ03MG6586",
    }
    is_valid, warnings, errors = validate_document_data(data, DocumentType.MOTOR_INSURANCE)
    assert is_valid is False
    assert any("Collision detected" in e for e in errors)
