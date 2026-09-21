"""
Field-specific cleaning and normalization rules.
Implements Rule 21, 22, 60, 61: Semantic field cleaning and no over-cleaning.
"""

import re
from typing import Optional


def clean_customer_name(raw_name: Optional[str]) -> Optional[str]:
    """
    Cleans raw customer/insured names.
    Examples:
    'MR. MEET KORAT' -> 'Mr Meet Korat'
    'MR MEET KORAT / 3654789/5' -> 'Mr Meet Korat'
    'MHL ABC MR MEET KORAT 3654789/5' -> 'Mr Meet Korat'
    """
    if not raw_name or not str(raw_name).strip():
        return None

    name = str(raw_name).strip()

    # If there are slashes separating an ID, take the name portion
    if "/" in name:
        parts = [p.strip() for p in name.split("/")]
        # Pick the part that contains alphabetic name characters without digits
        valid_name_part = None
        for p in parts:
            if re.search(r"[a-zA-Z]", p) and not re.search(r"\d{3,}", p):
                valid_name_part = p
                break
        if valid_name_part:
            name = valid_name_part
        else:
            name = parts[0]

    # Remove extraneous trailing numbers/IDs (e.g. '3654789/5' or '123456')
    name = re.sub(r"\s+\d+[\d/\-]*$", "", name)

    # Remove extraneous internal enterprise prefixes if name starts with corporate jargon like 'MHL ABC MR ...'
    # Match patterns like 'MHL ABC MR MEET KORAT'
    honorific_match = re.search(r"\b(mr|mrs|ms|dr|shri|smt|m/s)\.?\s+([a-zA-Z\s]+)", name, flags=re.IGNORECASE)
    if honorific_match:
        name = f"{honorific_match.group(1)} {honorific_match.group(2)}".strip()

    # Clean punctuation except valid name chars
    name = re.sub(r"[^\w\s\.\&]", "", name)

    # Convert to Title Case with proper honorific formatting
    words = name.split()
    cleaned_words = []
    for w in words:
        wl = w.lower().rstrip(".")
        if wl in ("mr", "mrs", "ms", "dr"):
            cleaned_words.append(wl.capitalize())
        elif wl in ("m/s",):
            cleaned_words.append("M/s")
        elif wl in ("shri", "smt"):
            cleaned_words.append(wl.capitalize())
        else:
            cleaned_words.append(w.capitalize())

    result = " ".join(cleaned_words).strip()
    return result if result else None


def clean_registration_number(raw_reg: Optional[str]) -> Optional[str]:
    """
    Normalizes vehicle registration numbers according to Indian RTO pattern.
    Example: 'GJ 03 MG 6586' -> 'GJ03MG6586'
    """
    if not raw_reg or not str(raw_reg).strip():
        return None

    # Remove all spaces and hyphens
    cleaned = re.sub(r"[\s\-\.]+", "", str(raw_reg)).upper()

    # Check if matches standard Indian vehicle pattern: State (2 letters) + RTO (1-2 digits) + Series (1-3 letters) + Number (1-4 digits)
    # E.g. GJ03MG6586 or DL1CAA1111 or MH12AB1234 or BH22AA1234A
    match = re.search(r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}", cleaned)
    if match:
        return match.group(0)

    # If already alphanumeric and reasonable length (6-12 chars), return uppercase
    if 6 <= len(cleaned) <= 12 and cleaned.isalnum():
        return cleaned

    return cleaned if cleaned else None


def clean_policy_number(raw_policy: Optional[str]) -> Optional[str]:
    """
    Normalizes policy number without destroying meaningful digits or delimiters.
    Example: 'P0023200023 / 4115 / 103739' -> 'P0023200023/4115/103739'
    Rule 61: Do NOT over-clean.
    """
    if not raw_policy or not str(raw_policy).strip():
        return None

    cleaned = str(raw_policy).strip()
    # Remove whitespace surrounding slashes or hyphens
    cleaned = re.sub(r"\s*([/\-])\s*", r"\1", cleaned)
    # Remove multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def clean_phone_number(raw_phone: Optional[str]) -> Optional[str]:
    """
    Normalizes Indian mobile/phone numbers to canonical format:
    Example: '+91-98765-43210' -> '+91 98765 43210'
    """
    if not raw_phone or not str(raw_phone).strip():
        return None

    digits = re.sub(r"[^\d]", "", str(raw_phone))
    if len(digits) == 10:
        return f"+91 {digits[:5]} {digits[5:]}"
    elif len(digits) == 12 and digits.startswith("91"):
        return f"+91 {digits[2:7]} {digits[7:]}"

    return str(raw_phone).strip()
