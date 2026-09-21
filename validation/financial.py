"""
Financial value parsing, Indian rupee formatting, and arithmetic reconciliation.
"""

import re
from typing import Optional, Union, Tuple
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


def validate_premium_components(
    own_damage: Optional[float],
    third_party: Optional[float],
    gst: Optional[float],
    total: Optional[float],
    tolerance: float = 5.0,
) -> Tuple[bool, Optional[str]]:
    """
    Validates arithmetic reconciliation of premium components.
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

    diff = abs(total - components_sum)
    if diff > tolerance:
        msg = f"Calculated premium sum ({format_currency_inr(components_sum)}) differs from declared total ({format_currency_inr(total)}) by ₹{diff:.2f}."
        logger.warning(msg)
        return False, msg

    return True, None
