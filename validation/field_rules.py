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

    # Reject standard document labels misidentified as names
    upper_raw = " ".join(name.upper().split()).rstrip(" :.-#")
    disallowed_labels = {
        "CUSTOMER ID", "CUSTOMER NO", "CUSTOMER NUMBER", "POLICY NUMBER", "POLICY NO",
        "PLAN NAME", "PRODUCT NAME", "ADDRESS", "PERMANENT ADDRESS", "CHASSIS", "CHASSIS NO",
        "CHASSIS NUMBER", "ENGINE NO", "ENGINE NUMBER", "REGISTRATION NO", "REGISTRATION NUMBER",
        "REG NO", "VEHICLE NO", "MAKE", "MODEL", "PERIOD OF INSURANCE",
        "PERSON", "CUSTOMER", "POLICYHOLDER", "POLICYHOLDER NAME", "POLICY HOLDER",
        "POLICY HOLDER NAME", "NAME", "NAME OF INSURED PERSON", "NAME OF INSURED",
        "NAME OF THE INSURED", "NAME OF PROPOSER", "INSURED PERSON", "INSURED PERSON(S)",
        "INSURED PERSONS", "SELF", "SPOUSE", "DAUGHTER", "SON", "FATHER", "MOTHER",
        "RELATIONSHIP", "RELATION", "DATE OF BIRTH", "DOB", "GENDER", "MALE", "FEMALE",
        "MEMBER", "MEMBER ID", "AGE"
    }
    if upper_raw in disallowed_labels:
        return None
    for dis in disallowed_labels:
        if upper_raw == dis or upper_raw.startswith(dis + ":") or upper_raw.startswith(dis + " "):
            if len(upper_raw.replace(dis, "").strip(" :.-#")) < 3:
                return None

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
    # Reject single letters (e.g. "M") or strings with fewer than 2 alphabetic characters
    if not result or len(re.sub(r"[^a-zA-Z]", "", result)) < 2:
        return None
    if result.upper() in disallowed_labels:
        return None
    return result


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

    # Reject plan names, labels, or generic text with no digits
    if not any(char.isdigit() for char in cleaned):
        return None

    # Reject obvious plan names even if followed by numbers (e.g. "SUPPORT PLUS", "GOLD PLAN")
    upper_c = cleaned.upper()
    if any(upper_c.startswith(pn) for pn in ["SUPPORT PLUS", "GOLD PLAN", "SILVER PLAN", "PLATINUM PLAN"]):
        return None

    # Remove whitespace surrounding slashes or hyphens
    cleaned = re.sub(r"\s*([/\-])\s*", r"\1", cleaned)
    # Remove multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if len(cleaned) < 4 or len(cleaned) > 45:
        return None

    return cleaned


def clean_engine_number(raw_engine: Optional[str]) -> Optional[str]:
    """
    Normalizes and strictly validates motor engine number.
    Rejects 'Chassis', 'Engine No.', 'Registration No.', and generic labels.
    """
    if not raw_engine or not str(raw_engine).strip():
        return None

    cleaned = re.sub(r"[\s\-\.]+", "", str(raw_engine)).upper()
    # Reject labels
    if cleaned in ("CHASSIS", "CHASSISNO", "CHASSISNUMBER", "ENGINENO", "ENGINENUMBER", "REGISTRATIONNO", "REGNO", "MODEL", "MAKE"):
        return None
    if any(cleaned.startswith(lbl) and len(cleaned) == len(lbl) for lbl in ["CHASSIS", "ENGINE", "REGN", "REGISTRATION"]):
        return None

    if len(cleaned) < 4 or len(cleaned) > 30:
        return None

    return cleaned


def clean_chassis_number(raw_chassis: Optional[str]) -> Optional[str]:
    """
    Normalizes and strictly validates motor chassis number (VIN).
    Rejects 'Chassis No.', 'Engine No.', 'Registration No.', and generic labels.
    """
    if not raw_chassis or not str(raw_chassis).strip():
        return None

    cleaned = re.sub(r"[\s\-\.]+", "", str(raw_chassis)).upper()
    # Reject labels
    if cleaned in ("CHASSIS", "CHASSISNO", "CHASSISNUMBER", "ENGINENO", "ENGINENUMBER", "REGISTRATIONNO", "REGNO", "MODEL", "MAKE"):
        return None
    if any(cleaned.startswith(lbl) and len(cleaned) == len(lbl) for lbl in ["CHASSIS", "ENGINE", "REGN", "REGISTRATION"]):
        return None

    if len(cleaned) < 5 or len(cleaned) > 30:
        return None

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


def clean_vehicle_class(raw_class: Optional[str]) -> Optional[str]:
    """
    Standardizes vehicle class to 1 of:
    (private car, commercial, two wheeler, miscellaneous, bus, 3 wheeler)
    """
    if not raw_class or not str(raw_class).strip():
        return None
    s = str(raw_class).strip().lower()
    if any(k in s for k in ["private car", "car", "4 wheeler", "motor car", "saloon"]):
        return "private car"
    elif any(k in s for k in ["two wheeler", "2 wheeler", "motorcycle", "scooter", "bike"]):
        return "two wheeler"
    elif any(k in s for k in ["3 wheeler", "three wheeler", "auto", "rickshaw"]):
        return "3 wheeler"
    elif "bus" in s:
        return "bus"
    elif any(k in s for k in ["commercial", "goods", "gcv", "pcv", "truck", "tractor", "lorry", "carrier"]):
        return "commercial"
    elif "miscellaneous" in s:
        return "miscellaneous"
    return str(raw_class).strip()


def clean_insurance_type(raw_type: Optional[str]) -> Optional[str]:
    """
    Standardizes insurance type to 1 of:
    (comprehensive/package, third party/liability, own damage)
    """
    if not raw_type or not str(raw_type).strip():
        return None
    s = str(raw_type).strip().lower()
    if any(k in s for k in ["comprehensive", "package", "bundled", "comp"]):
        return "comprehensive/package"
    elif any(k in s for k in ["third party", "third-party", "liability", "act only", "tp"]):
        return "third party/liability"
    elif any(k in s for k in ["own damage", "standalone od", "od only", "od"]):
        return "own damage"
    return str(raw_type).strip()


def clean_ncb(raw_ncb: Optional[str]) -> Optional[str]:
    """
    Standardizes NCB percentage or preserves monetary discount amount.
    Examples:
    '20' -> '20%', '20%' -> '20%', '0.0 %' -> '0.0%'
    'Rs. 500.00' -> '500.00', '1755.50' -> '1755.50'
    """
    if not raw_ncb or not str(raw_ncb).strip():
        return None
    s = str(raw_ncb).strip()

    # Percentage pattern
    m_pct = re.search(r"(\d{1,2}(?:\.\d{1,2})?)\s*%", s)
    if m_pct:
        return f"{m_pct.group(1)}%"

    # Explicit currency or decimal amount (e.g. Rs 500.00 or 1755.50)
    if any(sym in s for sym in ["Rs", "₹", "INR", "/-"]) or re.search(r"^\d+\.\d{2}$", s):
        from validation.financial import parse_currency_amount
        amt = parse_currency_amount(s)
        if amt is not None:
            return str(amt)

    # Standalone 1-2 digit integer (standard Indian NCB slabs: 0, 20, 25, 35, 45, 50)
    m_num = re.search(r"\b(\d{1,2})\b", s)
    if m_num:
        val = int(m_num.group(1))
        if val in (0, 20, 25, 35, 45, 50, 65):
            return f"{val}%"
        return f"{val}%"

    return s


def clean_seating_capacity(raw_cap: Optional[str]) -> Optional[str]:
    """Extracts digits only for seating capacity."""
    if not raw_cap or not str(raw_cap).strip():
        return None
    m = re.search(r"\b(\d{1,2})\b", str(raw_cap))
    return m.group(1) if m else str(raw_cap).strip()


def clean_year(raw_year: Optional[str]) -> Optional[str]:
    """Extracts 4-digit manufacturing year."""
    if not raw_year or not str(raw_year).strip():
        return None
    m = re.search(r"\b(19\d{2}|20\d{2})\b", str(raw_year))
    return m.group(1) if m else str(raw_year).strip()

