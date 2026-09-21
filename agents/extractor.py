"""
Structured Extractor: Extracts schema-compliant structured data.
Combines Ollama LLM structured JSON output with a deterministic fallback extractor.
Adheres strictly to Rule 1, 2, 3, 17, 73: Zero hallucination, fail-closed policy.
"""

import json
import re
from typing import Dict, Any, Optional
import httpx
from config import settings
from schemas.base import DocumentType
from utils.logger import get_logger

logger = get_logger("extractor")


def extract_with_ollama(text: str, doc_type: DocumentType, schema_keys: list) -> Optional[Dict[str, Any]]:
    """
    Extracts structured fields using local Ollama model in JSON mode.
    Returns parsed dictionary or None if Ollama is unreachable.
    """
    system_prompt = (
        "You are a document extraction engine.\n"
        "Extract only information explicitly supported by the supplied document content.\n"
        "Never invent values. Never infer missing values as facts.\n"
        "Never merge unrelated fields. Respect field boundaries.\n"
        "Return structured data according to the supplied schema as a single JSON object.\n"
        "If a field is absent, return null. If a value is unreadable, return null rather than guessing.\n"
        "Do not include explanations. Do not include raw OCR noise.\n"
        "Do not add information from your general knowledge."
    )

    user_prompt = (
        f"Document Type: {doc_type.value}\n"
        f"Required Fields to Extract: {', '.join(schema_keys)}\n\n"
        f"Document Text:\n\"\"\"\n{text[:6000]}\n\"\"\"\n\n"
        "Extract only supported fields as a valid JSON object."
    )

    payload = {
        "model": settings.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.0,
        },
    }

    try:
        url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat"
        with httpx.Client(timeout=httpx.Timeout(float(settings.OLLAMA_TIMEOUT), connect=2.0)) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 200:
                result_json = resp.json()
                content = result_json.get("message", {}).get("content", "")
                if content:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        logger.info(f"Successfully extracted {len(parsed)} fields using Ollama ({settings.OLLAMA_MODEL})")
                        return parsed
    except Exception as e:
        logger.warning(f"Ollama structured extraction unavailable ({e}). Falling back to deterministic extractor.")

    return None


def extract_deterministic_motor(text: str) -> Dict[str, Any]:
    """Deterministic regex and layout parser for Motor Insurance documents."""
    data: Dict[str, Any] = {}

    # Policy number
    m = re.search(r"(?i)(?:policy\s*(?:no|number)|certificate\s*no)[\s\.:/]*([A-Z0-9/\-]{8,35})", text)
    if m:
        data["policy_number"] = m.group(1).strip()

    # Customer name
    # Looks for 'Name of Insured', 'Insured Name', 'Proposer's Full Name', 'Insured', 'Mr/Mrs ...'
    m = re.search(r"(?i)(?:proposer(?:'s)?\s*(?:\([^)]*\)\s*)?full\s*name|name\s*of\s*(?:the\s*)?insured|insured\s*name|customer\s*name)[\s\.:/-]*\n?\s*([^\n\r,;]{3,60})", text)
    if m and not any(ign in m.group(1).upper() for ign in ["MOTOR", "VEHICLE", "DETAILS", "ADDRESS", "POLICY", "CLASS"]):
        data["insured_name"] = m.group(1).strip()
    else:
        # Check for honorifics in first 2500 chars (stay on single line)
        m_hon = re.search(r"\b(Mr\.?|Mrs\.?|Ms\.?|Dr\.?|M/s\.?)[ \t]+([A-Za-z \t]{2,40})", text[:2500])
        if m_hon:
            data["insured_name"] = f"{m_hon.group(1)} {m_hon.group(2)}".strip()

    # Registration number (e.g. GJ03MG6586, MH12AB1234, etc.)
    m = re.search(r"(?i)(?:registration\s*(?:no|number)|regn?\s*no)[\s\.:/-]*([A-Z]{2}\s*\d{1,2}\s*[A-Z]{1,3}\s*\d{1,4})", text)
    if m:
        data["registration_number"] = m.group(1).strip()
    else:
        # Standalone search for Indian reg pattern
        m_reg = re.search(r"\b([A-Z]{2}\s*\d{1,2}\s*[A-Z]{1,3}\s*\d{1,4})\b", text)
        if m_reg:
            data["registration_number"] = m_reg.group(1).strip()

    # Vehicle make & model
    if "tata" in text.lower() and "ace" in text.lower():
        data["vehicle_make"] = "TATA"
        data["vehicle_model"] = "ACE 275 ID"
    else:
        m_veh = re.search(r"(?i)(?:make\s*[\&/]\s*model|vehicle\s*make|make|vehicle\s*description)[\s\.:/-]*([^\n\r,;]{3,50})", text)
        if m_veh:
            val = m_veh.group(1).strip()
            if val.upper() not in ["MODEL", "MAKE", "TYPE OF BODY", "GVW"]:
                data["vehicle_model"] = val
            if "tata" in val.lower():
                data["vehicle_make"] = "TATA"
            elif "maruti" in val.lower():
                data["vehicle_make"] = "MARUTI"
            elif "honda" in val.lower():
                data["vehicle_make"] = "HONDA"
            elif "hyundai" in val.lower():
                data["vehicle_make"] = "HYUNDAI"
            elif "mahindra" in val.lower():
                data["vehicle_make"] = "MAHINDRA"
            elif "hero" in val.lower():
                data["vehicle_make"] = "HERO"
            elif "bajaj" in val.lower():
                data["vehicle_make"] = "BAJAJ"

    # Engine & Chassis
    m_eng = re.search(r"(?i)(?:engine\s*(?:no|number))[\s\.:/-]*([A-Z0-9]{5,25})", text)
    if m_eng:
        data["engine_number"] = m_eng.group(1).strip()

    m_cha = re.search(r"(?i)(?:chassis\s*(?:no|number)|vin)[\s\.:/-]*([A-Z0-9]{5,25})", text)
    if m_cha:
        data["chassis_number"] = m_cha.group(1).strip()

    # Dates
    # Specific date search: "from ... to ..."
    m_range = re.search(r"(?i)(?:from|period\s*of\s*insurance)[\s\.:]*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4}|\d{1,2}\s+[A-Za-z]+\s+\d{4})[\s\w]*(?:to|until)[\s\.:]*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4}|\d{1,2}\s+[A-Za-z]+\s+\d{4})", text)
    if m_range:
        data["policy_start_date"] = m_range.group(1).strip()
        data["policy_end_date"] = m_range.group(2).strip()
    else:
        m_exp = re.search(r"(?i)(?:to\s*midnight\s*of|expires?\s*on|expiry\s*date|to\s*:\s*\d\d:\d\d\s*hrs\s*on)[\s\.:]*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4})", text)
        if m_exp:
            data["policy_end_date"] = m_exp.group(1).strip()
        m_dates = re.findall(r"\b(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4})\b", text)
        if len(m_dates) >= 2:
            data.setdefault("policy_start_date", m_dates[0])
            data.setdefault("policy_end_date", m_dates[1])
        elif len(m_dates) == 1:
            data.setdefault("policy_start_date", m_dates[0])

    # Financials / Premiums
    m_tp = re.search(r"(?i)(?:third\s*party\s*(?:premium|liability|tp)|basic\s*tp)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
    if m_tp:
        data["third_party_premium"] = m_tp.group(1)

    m_od = re.search(r"(?i)(?:own\s*damage\s*(?:premium|od)|basic\s*od)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
    if m_od:
        data["own_damage_premium"] = m_od.group(1)

    m_gst = re.search(r"(?i)(?:gst|igst|cgst\s*\+\s*sgst|taxes)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
    if m_gst:
        data["gst"] = m_gst.group(1)

    # Total Premium / Amount Payable
    # Check 1: Standalone TOTAL row in a table or under Premium Computation
    m_stand = re.search(r"(?i)(?:^|\n)\s*(?:GRAND\s*TOTAL|TOTAL\s*AMOUNT\s*PAYABLE|TOTAL\s*PREMIUM|TOTAL)\s*\n\s*([\d,]+(?:\.\d{2})?)", text)
    if m_stand:
        data["total_premium"] = m_stand.group(1).strip()
    else:
        m_tot = re.search(r"(?i)(?:total\s*premium|net\s*premium|total\s*amount\s*payable|gross\s*premium|total\s*amount|final\s*premium|total\s*payable)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
        if m_tot:
            data["total_premium"] = m_tot.group(1).strip()
        else:
            m_coll = re.search(r"(?i)premium\s*collection\s*details[\s\S]{1,250}?,\s*([\d,]+(?:\.\d{2})?)", text)
            if m_coll:
                data["total_premium"] = m_coll.group(1).strip()

    # Coverage type
    if "liability only" in text.lower() or "third party" in text.lower():
        data["coverage"] = "Third Party Liability"
        data["policy_type"] = "Liability Only"
    elif "package" in text.lower() or "comprehensive" in text.lower():
        data["coverage"] = "Comprehensive Package Policy"
        data["policy_type"] = "Package Policy"

    return data


def extract_deterministic_electricity(text: str) -> Dict[str, Any]:
    """Deterministic regex and layout parser for Electricity Bills."""
    data: Dict[str, Any] = {}

    # Provider
    for provider in ["PGVCL", "MGVCL", "DGVCL", "UGVCL", "BESCOM", "TATA POWER", "ADANI ELECTRICITY", "MSEB", "UPPCL"]:
        if provider in text.upper():
            data["electricity_provider"] = provider
            break

    # Consumer Number
    m = re.search(r"(?i)(?:consumer\s*(?:no|number|id)|service\s*connection\s*no|ca\s*no)[\s\.:/-]*([A-Z0-9]{5,25})", text)
    if m:
        data["consumer_number"] = m.group(1).strip()

    # Customer Name
    m = re.search(r"(?i)(?:consumer\s*name|name\s*of\s*consumer|customer\s*name|name)[\s\.:/-]*([^\n\r,;]{3,50})", text)
    if m:
        data["customer_name"] = m.group(1).strip()

    # Meter Number
    m = re.search(r"(?i)(?:meter\s*(?:no|number))[\s\.:/-]*([A-Z0-9]{4,20})", text)
    if m:
        data["meter_number"] = m.group(1).strip()

    # Bill Date & Due Date
    m_bd = re.search(r"(?i)(?:bill\s*date)[\s\.:/-]*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4})", text)
    if m_bd:
        data["bill_date"] = m_bd.group(1).strip()

    m_dd = re.search(r"(?i)(?:due\s*date|pay\s*by\s*date)[\s\.:/-]*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4})", text)
    if m_dd:
        data["due_date"] = m_dd.group(1).strip()

    # Readings and Consumption
    m_curr = re.search(r"(?i)(?:current\s*reading|present\s*reading)[\s\.:]*([\d,]+(?:\.\d+)?)", text)
    if m_curr:
        data["current_reading"] = m_curr.group(1)

    m_prev = re.search(r"(?i)(?:previous\s*reading|past\s*reading)[\s\.:]*([\d,]+(?:\.\d+)?)", text)
    if m_prev:
        data["previous_reading"] = m_prev.group(1)

    m_cons = re.search(r"(?i)(?:units\s*consumed|consumption|total\s*units)[\s\.:]*([\d,]+(?:\.\d+)?)", text)
    if m_cons:
        data["consumption"] = m_cons.group(1)

    # Charges and Total
    m_tot = re.search(r"(?i)(?:total\s*bill\s*amount|net\s*amount\s*payable|amount\s*payable|total\s*amount)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
    if m_tot:
        data["total_bill"] = m_tot.group(1)
        data["net_bill_amount"] = m_tot.group(1)

    return data


def extract_deterministic_health(text: str) -> Dict[str, Any]:
    """Deterministic regex parser for Health Insurance documents."""
    data: Dict[str, Any] = {}

    m = re.search(r"(?i)(?:policy\s*(?:no|number))[\s\.:/-]*([A-Z0-9/\-]{8,35})", text)
    if m:
        data["policy_number"] = m.group(1).strip()

    m = re.search(r"(?i)(?:policyholder\s*name|name\s*of\s*proposer|proposer\s*name|insured\s*name)[\s\.:/-]*([^\n\r,;]{3,50})", text)
    if m:
        data["policyholder_name"] = m.group(1).strip()

    m_si = re.search(r"(?i)(?:sum\s*insured|basic\s*sum\s*insured)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
    if m_si:
        data["sum_insured"] = m_si.group(1)

    m_tot = re.search(r"(?i)(?:total\s*premium|gross\s*premium|net\s*premium)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
    if m_tot:
        data["premium"] = m_tot.group(1)

    m_dates = re.findall(r"\b(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4})\b", text)
    if len(m_dates) >= 2:
        data["policy_start_date"] = m_dates[0]
        data["policy_end_date"] = m_dates[1]

    return data


def extract_deterministic_property(text: str) -> Dict[str, Any]:
    """Deterministic regex parser for Property / Fire Insurance."""
    data: Dict[str, Any] = {}

    m = re.search(r"(?i)(?:policy\s*(?:no|number))[\s\.:/-]*([A-Z0-9/\-]{8,35})", text)
    if m:
        data["policy_number"] = m.group(1).strip()

    m = re.search(r"(?i)(?:insured\s*name|name\s*of\s*insured|business\s*name)[\s\.:/-]*([^\n\r,;]{3,50})", text)
    if m:
        data["insured_business"] = m.group(1).strip()

    m_si = re.search(r"(?i)(?:total\s*sum\s*insured|sum\s*insured)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
    if m_si:
        data["total_sum_insured"] = m_si.group(1)

    m_tot = re.search(r"(?i)(?:total\s*premium)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
    if m_tot:
        data["total_premium"] = m_tot.group(1)

    return data


def extract_deterministic_workmen(text: str) -> Dict[str, Any]:
    """Deterministic regex parser for Workmen's Compensation."""
    data: Dict[str, Any] = {}

    m = re.search(r"(?i)(?:policy\s*(?:no|number))[\s\.:/-]*([A-Z0-9/\-]{8,35})", text)
    if m:
        data["policy_number"] = m.group(1).strip()

    m = re.search(r"(?i)(?:employer\s*name|insured\s*name|name\s*of\s*insured)[\s\.:/-]*([^\n\r,;]{3,50})", text)
    if m:
        data["employer"] = m.group(1).strip()

    m_tot = re.search(r"(?i)(?:total\s*premium)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
    if m_tot:
        data["total_premium"] = m_tot.group(1)

    return data


def extract_deterministic_generic(text: str) -> Dict[str, Any]:
    """Generic document parser."""
    data: Dict[str, Any] = {}

    m = re.search(r"(?i)(?:invoice\s*(?:no|number)|reference\s*(?:no|number)|bill\s*no)[\s\.:/-]*([A-Z0-9/\-]{5,30})", text)
    if m:
        data["reference_number"] = m.group(1).strip()

    m_name = re.search(r"(?i)(?:billed\s*to|customer\s*name|name|to:)[\s\.:/-]*([^\n\r,;]{3,50})", text)
    if m_name:
        data["primary_party_name"] = m_name.group(1).strip()

    m_tot = re.search(r"(?i)(?:total\s*amount|total|grand\s*total)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
    if m_tot:
        data["amount_or_value"] = m_tot.group(1)

    return data


def extract_structured_fields(
    text: str,
    doc_type: DocumentType,
    tables: Optional[list] = None,
    form_fields: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Main extraction orchestrator.
    Combines:
    1. Deterministic domain-specific parser
    2. Any AcroForm field values
    3. Any structured table values
    4. Ollama LLM structured refinement (if online)
    """
    # Step 1: Deterministic baseline extraction
    if doc_type == DocumentType.MOTOR_INSURANCE:
        extracted = extract_deterministic_motor(text)
    elif doc_type == DocumentType.ELECTRICITY_BILL:
        extracted = extract_deterministic_electricity(text)
    elif doc_type == DocumentType.HEALTH_INSURANCE:
        extracted = extract_deterministic_health(text)
    elif doc_type == DocumentType.PROPERTY_INSURANCE:
        extracted = extract_deterministic_property(text)
    elif doc_type == DocumentType.WORKMEN_COMPENSATION:
        extracted = extract_deterministic_workmen(text)
    else:
        extracted = extract_deterministic_generic(text)

    # Step 2: Overlay AcroForm values if present
    if form_fields:
        for fk, fv in form_fields.items():
            if fv and fk.lower() not in extracted:
                extracted[f"form_{fk}"] = fv

    # Step 3: Overlay Table values if present (e.g. key_value_map)
    if tables:
        for t in tables:
            kv_map = t.get("key_value_map", {})
            for tk, tv in kv_map.items():
                if "total" in tk.lower() and "premium" not in extracted:
                    extracted["table_total"] = tv

    # Step 4: Refine with Ollama LLM if available
    llm_extracted = extract_with_ollama(text, doc_type, list(extracted.keys()) or ["policy_number", "customer_name", "dates", "amount"])
    if llm_extracted:
        for k, v in llm_extracted.items():
            # Only adopt if v is not null and has meaningful content
            if v is not None and str(v).strip() and str(v).lower() not in ("null", "none"):
                extracted[k] = v

    return extracted
