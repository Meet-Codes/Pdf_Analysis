"""
Deterministic date normalization and temporal validation.
"""

import re
from typing import Optional, Tuple
from datetime import datetime
from dateutil import parser as date_parser
from utils.logger import get_logger

logger = get_logger("validation.dates")


def parse_and_normalize_date(date_raw: Optional[str]) -> Optional[str]:
    """
    Normalizes any standard date string into canonical 'DD Month YYYY' format.
    Example: '29/03/2023' -> '29 March 2023'.
    Returns None if date cannot be parsed reliably.
    """
    if not date_raw or not str(date_raw).strip():
        return None

    cleaned = str(date_raw).strip()
    # Remove ordinal suffixes like 29th, 1st, 2nd
    cleaned = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", cleaned, flags=re.IGNORECASE)

    # First attempt dayfirst=True parsing
    try:
        dt = date_parser.parse(cleaned, dayfirst=True)
        # Sanity check year
        if 1900 <= dt.year <= 2100:
            return dt.strftime("%d %B %Y").lstrip("0")
    except Exception:
        pass

    # Regex fallback for DD/MM/YYYY or DD-MM-YYYY
    match = re.search(r"\b(\d{1,2})[-/\.](\d{1,2})[-/\.](\d{4})\b", cleaned)
    if match:
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        try:
            dt = datetime(year, month, day)
            return dt.strftime("%d %B %Y").lstrip("0")
        except ValueError:
            pass

    return None


def validate_date_range(start_date_str: Optional[str], end_date_str: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    Validates that start_date <= end_date.
    Returns (is_valid, error_message_or_none)
    """
    if not start_date_str or not end_date_str:
        return True, None

    try:
        dt_start = date_parser.parse(start_date_str, dayfirst=True)
        dt_end = date_parser.parse(end_date_str, dayfirst=True)

        if dt_start > dt_end:
            return False, f"Start date ({start_date_str}) cannot be later than end date ({end_date_str})."
        return True, None
    except Exception as e:
        logger.debug(f"Could not compare dates '{start_date_str}' and '{end_date_str}': {e}")
        return True, None
