"""
Financial value parsing, Indian rupee formatting, and arithmetic reconciliation.
Strict non-destructive policy: Never overwrite declared document totals.
"""

import re
from typing import Optional, Union, Tuple, Dict, Any
from utils.formatting import format_currency_inr
from utils.logger import get_logger

logger = get_logger("validation.financial")


def parse_currency_amount(raw_val: Union[str, float, int, None]) -> Optional[float]:
    """
    Parses messy financial strings into a clean float.
    Examples:
    'Rs. 15,430/-' -> 15430.0
    '₹ 1,25,000' -> 125000.0
    'INR 18,207.00' -> 18207.0
    '15430.0000' -> 15430.0
    """
    if raw_val is None:
        return None

    if isinstance(raw_val, (int, float)):
        return float(raw_val)

    s = str(raw_val).strip()
    if not s:
        return None

    # Remove currency prefixes, suffixes, spaces
    cleaned = re.sub(r"(?i)rs\.?|inr|₹|/-", "", s)
    # Remove commas
    cleaned = cleaned.replace(",", "").strip()

    # Extract float number
    match = re.search(r"[-+]?\d*\.?\d+", cleaned)
    if match:
        try:
            return round(float(match.group(0)), 2)
        except ValueError:
            return None

    return None


def validate_full_premium_breakdown(
    basic_premium: Optional[float],
    add_ons: Optional[float],
    discounts: Optional[float],
    net_premium: Optional[float],
    taxes: Optional[float],
    total_declared: Optional[float],
    tolerance: float = 5.0,
) -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Validates arithmetic reconciliation of all premium components:
    1. Basic Premium + Add-ons - Discounts ≈ Net Premium (if present)
    2. (Net Premium or Basic Premium + Add-ons - Discounts) + Taxes ≈ Total Declared
    IMPORTANT: Never overwrite declared document total merely because arithmetic differs!
    If there is a discrepancy, return is_valid=False with an informative warning string and calculated total.
    """
    if total_declared is None:
        return True, None, None

    # Calculate base/net component
    calculated_net = 0.0
    has_net_info = False

    if net_premium is not None:
        calculated_net = net_premium
        has_net_info = True
    elif basic_premium is not None:
        calculated_net = basic_premium
        if add_ons is not None:
            calculated_net += add_ons
        if discounts is not None:
            calculated_net -= discounts
        has_net_info = True

    if not has_net_info:
        return True, None, total_declared

    calculated_total = calculated_net
    if taxes is not None:
        calculated_total += taxes

    calculated_total = round(calculated_total, 2)
    diff = abs(total_declared - calculated_total)

    if diff > tolerance:
        msg = (
            f"Financial arithmetic discrepancy: Calculated total ({format_currency_inr(calculated_total)}) "
            f"differs from declared document total ({format_currency_inr(total_declared)}) by ₹{diff:.2f}. "
            "Declared document total is retained as authoritative."
        )
        logger.warning(msg)
        return False, msg, calculated_total

    return True, None, calculated_total


def validate_premium_components(
    own_damage: Optional[float],
    third_party: Optional[float],
    gst: Optional[float],
    total: Optional[float],
    tolerance: float = 5.0,
) -> Tuple[bool, Optional[str]]:
    """
    Validates arithmetic reconciliation of motor premium components.
    Total ≈ (Own Damage + Third Party) + GST
    """
    if total is None:
        return True, None

    components_sum = 0.0
    has_components = False

    if own_damage is not None:
        components_sum += own_damage
        has_components = True

    if third_party is not None:
        components_sum += third_party
        has_components = True

    if not has_components:
        return True, None

    if gst is not None:
        components_sum += gst

    components_sum = round(components_sum, 2)
    diff = abs(total - components_sum)
    if diff > tolerance:
        msg = (
            f"Calculated premium sum ({format_currency_inr(components_sum)}) "
            f"differs from declared total ({format_currency_inr(total)}) by ₹{diff:.2f}."
        )
        logger.warning(msg)
        return False, msg

    return True, None
