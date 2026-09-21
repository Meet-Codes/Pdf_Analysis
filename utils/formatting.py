"""
Formatting utilities for financial numbers, dates, and clean human-readable output.
Strict adherence to Rule 15: No hallucinations, clean deterministic canonical representations.
"""

import re
from typing import Any, Optional, Union
from datetime import datetime
from dateutil import parser as date_parser


def format_currency_inr(amount: Union[float, int, str, None]) -> str:
    """
    Format a numeric amount into Indian Currency notation (e.g. ₹15,430 or ₹1,25,000).
    Never prints 15430.0000 or raw floats unless explicitly required.
    """
    if amount is None:
        return "Not available in the document"

    if isinstance(amount, str):
        # Extract numeric content
        cleaned = re.sub(r"[^\d.-]", "", amount.replace(",", ""))
        if not cleaned:
            return "Not available in the document"
        try:
            val = float(cleaned)
        except ValueError:
            return amount.strip()
    else:
        val = float(amount)

    # Indian comma grouping algorithm
    is_negative = val < 0
    val = abs(val)
    # Check if there are significant decimals
    has_decimals = (val % 1) >= 0.005
    int_part = int(val)
    dec_part = f"{(val % 1):.2f}"[1:] if has_decimals else ""

    s = str(int_part)
    if len(s) <= 3:
        formatted_int = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        # Group rest by 2 from right to left
        groups = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        formatted_int = ",".join(groups) + "," + last3

    res = f"₹{formatted_int}{dec_part}"
    return f"-{res}" if is_negative else res


def format_canonical_date(date_str: Optional[str]) -> str:
    """
    Normalizes arbitrary dates (e.g., '29/03/2023', '2023-03-29', '29-Mar-2023')
    into canonical format: '29 March 2023'.
    If unparseable, returns original cleaned or 'Not available in the document'.
    """
    if not date_str or not str(date_str).strip():
        return "Not available in the document"

    date_str = str(date_str).strip()
    # Normalize slashes and dots
    norm_str = re.sub(r"[\./]", "-", date_str)

    # Common Indian date formats: DD-MM-YYYY
    try:
        # Check DD-MM-YYYY specifically if 3 parts
        parts = norm_str.split("-")
        if len(parts) == 3 and len(parts[0]) <= 2 and len(parts[2]) == 4:
            day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
            dt = datetime(year, month, day)
            return dt.strftime("%d %B %Y").lstrip("0")
    except Exception:
        pass

    try:
        dt = date_parser.parse(date_str, dayfirst=True)
        return dt.strftime("%d %B %Y").lstrip("0")
    except Exception:
        return date_str


def clean_display_value(value: Any) -> str:
    """
    Standard production-grade null and display value cleaner.
    Guarantees that None, 'null', 'None', 'N/A', 'Unknown', '???' are NEVER shown.
    """
    if value is None:
        return "Not available in the document"
    s = str(value).strip()
    if s.lower() in ("none", "null", "n/a", "na", "unknown", "???", "", "-", "undefined"):
        return "Not available in the document"
    return s
