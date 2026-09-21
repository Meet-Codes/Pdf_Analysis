"""
Validation Engine: Cross-field validation, mathematical reconciliation, and integrity checks.
Implements Rule 23, 24: Detects contradictions or impossible values internally.
"""

from typing import Dict, Any, List, Tuple
from schemas.base import DocumentType
from validation.dates import validate_date_range
from validation.financial import validate_premium_components
from utils.logger import get_logger

logger = get_logger("validator")


def validate_document_data(data: Dict[str, Any], doc_type: DocumentType) -> Tuple[bool, List[str], List[str]]:
    """
    Executes domain-specific cross-field validations on normalized data.
    Returns: (is_valid, list_of_warnings, list_of_errors)
    """
    warnings: List[str] = []
    errors: List[str] = []

    # 1. Universal Date Range Validation
    start_date = data.get("policy_start_date") or data.get("bill_date") or data.get("date_of_issue")
    end_date = data.get("policy_end_date") or data.get("due_date") or data.get("expiry_or_due_date")

    if start_date and end_date:
        valid_range, err_msg = validate_date_range(start_date, end_date)
        if not valid_range:
            errors.append(err_msg)

    # 2. Motor Insurance Validations
    if doc_type == DocumentType.MOTOR_INSURANCE:
        reg_no = data.get("registration_number")
        eng_no = data.get("engine_number")
        cha_no = data.get("chassis_number")

        # Cross-field collision check: Registration should not accidentally equal Engine/Chassis
        if reg_no and eng_no and reg_no == eng_no:
            errors.append("Collision detected: Registration number cannot be identical to Engine number.")
        if reg_no and cha_no and reg_no == cha_no:
            errors.append("Collision detected: Registration number cannot be identical to Chassis number.")

        # Premium reconciliation
        od = data.get("own_damage_premium")
        tp = data.get("third_party_premium")
        gst = data.get("gst")
        total = data.get("total_premium")
        valid_prem, prem_msg = validate_premium_components(od, tp, gst, total)
        if not valid_prem and prem_msg:
            warnings.append(prem_msg)

    # 3. Electricity Bill Validations
    elif doc_type == DocumentType.ELECTRICITY_BILL:
        raw_prev = data.get("previous_reading")
        raw_curr = data.get("current_reading")
        raw_cons = data.get("consumption")

        try:
            prev_rdg = float(raw_prev) if raw_prev is not None else None
            curr_rdg = float(raw_curr) if raw_curr is not None else None
            consumption = float(raw_cons) if raw_cons is not None else None
        except (ValueError, TypeError):
            prev_rdg, curr_rdg, consumption = None, None, None

        if prev_rdg is not None and curr_rdg is not None:
            if curr_rdg < prev_rdg:
                warnings.append(f"Current meter reading ({curr_rdg}) is less than previous reading ({prev_rdg}). Possible meter rollover.")
            else:
                expected_consumption = curr_rdg - prev_rdg
                if consumption is not None and abs(consumption - expected_consumption) > 1.0:
                    warnings.append(f"Calculated consumption ({expected_consumption}) differs from bill consumption ({consumption}).")

    # 4. Property Insurance Validations
    elif doc_type == DocumentType.PROPERTY_INSURANCE:
        base = data.get("base_premium")
        terr = data.get("terrorism_premium")
        gst = data.get("gst_amount")
        total = data.get("total_premium")
        if base is not None or terr is not None:
            combined_base = (base or 0.0) + (terr or 0.0)
            valid_prem, prem_msg = validate_premium_components(combined_base, 0.0, gst, total)
            if not valid_prem and prem_msg:
                warnings.append(prem_msg)

    is_valid = len(errors) == 0
    return is_valid, warnings, errors
