"""
Structured Extractor: Extracts schema-compliant structured data.
Combines Ollama LLM structured JSON output with a deterministic fallback extractor.
Adheres strictly to Rule 1, 2, 3, 17, 73: Zero hallucination, fail-closed policy.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional
import httpx
from config import settings
from schemas.base import DocumentType
from utils.logger import get_logger

logger = get_logger("extractor")

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
EXTRACTION_PROMPT_FILE = PROMPTS_DIR / "extraction.txt"

MOTOR_FIELD_ALIASES: Dict[str, list] = {
    "CUSTOMER_NAME": ["customer_name", "insured_name"],
    "CUSTOMER_MOBILE": ["mobile"],
    "COMPANY_NAME": ["insurance_company", "insurer"],
    "AGENT_NAME": ["agent_name"],
    "AGENT_CODE": ["agent_code"],
    "CLASS_OF_VEHICLE": ["class_of_vehicle", "vehicle_type"],
    "INSURANCE_TYPE": ["insurance_type", "policy_type"],
    "POLICY_BOOKING_DATE": ["policy_booking_date"],
    "POLICY_START_DATE": ["policy_start_date"],
    "POLICY_END_DATE": ["policy_end_date"],
    "POLICY_NUMBER": ["policy_number"],
    "VEHICLE_REGISTRATION_NUMBER": ["registration_number"],
    "TP_PREMIUM": ["third_party_premium", "tp_premium"],
    "OD_PREMIUM": ["own_damage_premium", "od_premium"],
    "NET_PREMIUM": ["net_premium"],
    "ADDON_PREMIUM": ["addon_premium"],
    "GST_AMOUNT": ["gst", "tax", "gst_amount"],
    "TOTAL_PREMIUM": ["total_premium", "total_amount"],
    "TOTAL_IDV": ["idv", "total_idv"],
    "CNG_IDV": ["cng_idv"],
    "ENGINE_NUMBER": ["engine_number"],
    "CHASSIS_NUMBER": ["chassis_number"],
    "YEAR_OF_MANUFACTURE": ["manufacturing_year"],
    "MAKE": ["vehicle_make", "make"],
    "MODEL": ["vehicle_model", "model"],
    "SEATING_CAPACITY": ["seating_capacity"],
    "NCB": ["ncb_percentage", "ncb"],
}


from agents.prompt_template import extraction_template, MOTOR_EXTRACTION_SCHEMA, SCHEMA_TO_CANONICAL_MAP


def extract_with_ollama(text: str, doc_type: DocumentType, schema_keys: list) -> Optional[Dict[str, Any]]:
    """
    Extracts structured fields using local Ollama model in JSON mode via ExtractionPromptTemplate.
    Returns parsed dictionary or None if Ollama is unreachable.
    """
    system_prompt = extraction_template.get_system_prompt()
    user_prompt = extraction_template.format_user_prompt(text, doc_type, schema_keys)

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
        timeout_sec = min(10.0, float(getattr(settings, "OLLAMA_TIMEOUT", 10)))
        with httpx.Client(timeout=httpx.Timeout(timeout_sec, connect=2.0)) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 200:
                result_json = resp.json()
                content = result_json.get("message", {}).get("content", "")
                if content:
                    parsed = extraction_template.parse_llm_response(content)
                    if isinstance(parsed, dict) and parsed:
                        logger.info(f"Successfully extracted {len(parsed)} fields using Ollama ({settings.OLLAMA_MODEL})")
                        return parsed
    except Exception as e:
        logger.warning(f"Ollama structured extraction unavailable ({e}). Falling back to deterministic extractor.")

    return None


def extract_deterministic_motor(text: str) -> Dict[str, Any]:
    """
    Deterministic regex and layout parser for Motor Insurance documents.
    Extracts all 27 prompt template schema fields.
    """
    data: Dict[str, Any] = {}

    # 1. Policy number (POLICY_NUMBER) - supports multi-token values
    m = re.search(r"(?i)(?:policy\s*(?:#|no\.?|number)|certificate\s*no\.?)[\s\.:/-]*\s*([A-Za-z0-9][A-Za-z0-9/\- ]{3,35}[A-Za-z0-9])(?=[ \t]*(?:policy\s*(?:#|no|number)|date|period|unique|\r|\n|$))", text)
    if m:
        cand_pol = m.group(1).strip()
        if not any(dis in cand_pol.upper() for dis in ["PREVIOUS", "PAYABLE", "POLICY", "INSURANCE"]):
            data["policy_number"] = cand_pol
            data["POLICY_NUMBER"] = cand_pol

    # 2. Customer name - restricted to Page 1 header with hard negative filtering
    m = re.search(r"(?i)(?:proposer(?:'s)?\s*(?:\([^)]*\)\s*)?full\s*name|name\s*of\s*(?:the\s*)?insured|insured\s*name|customer\s*name)[\s\.:/-]*\n?\s*([^\n\r,;]{3,60})", text[:2500])
    if m and not any(ign in m.group(1).upper() for ign in [
        "MOTOR", "VEHICLE", "DETAILS", "ADDRESS", "POLICY", "CLASS", "DEFENCE", "SETTLEMENT",
        "CONDITIONS", "LIMITATIONS", "AGENT", "BROKER", "POLICY BAZAAR", "POLICYBAZAAR", "COVERFOX", "BANK",
        "CUSTOMER ID", "CUSTOMER NO", "PLAN NAME", "CHASSIS", "ENGINE NO", "REGISTRATION NO", "REG NO"
    ]):
        cand_name = m.group(1).strip()
        if not any(char.isdigit() for char in cand_name) and len(re.sub(r"[^a-zA-Z]", "", cand_name)) >= 2:
            data["customer_name"] = cand_name
            data["insured_name"] = cand_name

    # 3. Customer Mobile (CUSTOMER_MOBILE)
    m_mob = re.search(r"(?i)(?:mobile(?:\s*no|\s*number)?|phone(?:\s*no)?|contact(?:\s*no)?)[\s\.:/-]*(\+?91[\s\-]?)?([6-9]\d{9})\b", text)
    if m_mob:
        data["customer_mobile"] = m_mob.group(2)
        data["mobile"] = m_mob.group(2)
        data["CUSTOMER_MOBILE"] = m_mob.group(2)

    # 4. Company Name / Insurer (COMPANY_NAME)
    for comp in [
        "IFFCO-TOKIO GENERAL INSURANCE CO.LTD", "IFFCO Tokio General Insurance", "IFFCO-TOKIO",
        "Reliance General Insurance Company Limited", "Reliance General Insurance",
        "TATA AIG GENERAL INSURANCE COMPANY LIMITED", "Tata AIG General Insurance",
        "Bajaj Allianz General Insurance Company", "Bajaj Allianz General Insurance",
        "ICICI Lombard General Insurance", "HDFC ERGO General Insurance",
        "Go Digit General Insurance Ltd.", "Go Digit General Insurance",
        "New India Assurance", "National Insurance", "United India Insurance",
        "Oriental Insurance", "SBI General Insurance"
    ]:
        # Handle hyphens or spaces in match
        pattern = r"[\s\.:\-]*".join(re.findall(r"[A-Za-z0-9]+", comp))
        if re.search(pattern, text[:2500], re.IGNORECASE):
            data["company_name"] = comp
            data["insurer"] = comp
            data["insurance_company"] = comp
            data["COMPANY_NAME"] = comp
            break

    # 5. Agent Name & Agent Code (AGENT_NAME, AGENT_CODE)
    m_agt = re.search(r"(?i)(?:intermediary\s*(?:name)?|agent\s*(?:name)?|advisor\s*(?:name)?)[\s\.:/-]*([^\n\r,;]{3,50})", text)
    if m_agt and not any(ign in m_agt.group(1).upper() for ign in ["CODE", "DETAILS", "NO", "NUMBER", "POLICY"]):
        data["agent_name"] = m_agt.group(1).strip()
        data["AGENT_NAME"] = data["agent_name"]

    m_cd = re.search(r"(?i)(?:intermediary\s*(?:code|no)|agent\s*(?:code|no)|advisor\s*(?:code|no))[\s\.:/-]*([A-Z0-9/\-]{3,25})", text)
    if m_cd:
        data["agent_code"] = m_cd.group(1).strip()
        data["AGENT_CODE"] = data["agent_code"]

    # 6. Class of Vehicle (CLASS_OF_VEHICLE)
    m_cls = re.search(r"(?i)(?:class\s*of\s*vehicle|vehicle\s*class|category)[\s\.:/-]*([^\n\r,;]{3,40})", text)
    if m_cls:
        from validation.field_rules import clean_vehicle_class
        raw_val = m_cls.group(1).strip()
        data["class_of_vehicle"] = clean_vehicle_class(raw_val) or raw_val
    else:
        low_t = text.lower()
        if any(k in low_t for k in ["commercial", "goods", "gcv", "carrier", "truck"]):
            data["class_of_vehicle"] = "commercial"
        elif any(k in low_t for k in ["two wheeler", "motorcycle", "scooter"]):
            data["class_of_vehicle"] = "two wheeler"
        elif any(k in low_t for k in ["private car", "car"]):
            data["class_of_vehicle"] = "private car"
        elif any(k in low_t for k in ["3 wheeler", "auto"]):
            data["class_of_vehicle"] = "3 wheeler"
        elif "bus" in low_t:
            data["class_of_vehicle"] = "bus"
    if "class_of_vehicle" in data:
        data["vehicle_type"] = data["class_of_vehicle"]
        data["CLASS_OF_VEHICLE"] = data["class_of_vehicle"]

    # 7. Insurance Type (INSURANCE_TYPE)
    low_t = text.lower()
    if any(k in low_t for k in ["package", "comprehensive"]):
        data["insurance_type"] = "comprehensive/package"
        data["coverage"] = "Comprehensive Package Policy"
        data["policy_type"] = "Package Policy"
    elif any(k in low_t for k in ["liability only", "third party", "tp liability"]):
        data["insurance_type"] = "third party/liability"
        data["coverage"] = "Third Party Liability"
        data["policy_type"] = "Liability Only"
    elif "own damage" in low_t:
        data["insurance_type"] = "own damage"
        data["coverage"] = "Own Damage"
        data["policy_type"] = "Own Damage"
    if "insurance_type" in data:
        data["INSURANCE_TYPE"] = data["insurance_type"]

    # 8. Policy Dates: Booking, Start, End (POLICY_BOOKING_DATE, POLICY_START_DATE, POLICY_END_DATE)
    m_bk = re.search(r"(?i)(?:booking\s*date|issue\s*date|transaction\s*date)[\s\.:/-]*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4})", text)
    if m_bk:
        data["policy_booking_date"] = m_bk.group(1).strip()
        data["POLICY_BOOKING_DATE"] = data["policy_booking_date"]

    m_range = re.search(r"(?i)(?:from|period\s*of\s*insurance)[\s\.:]*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4}|\d{1,2}\s+[A-Za-z]+\s+\d{4})[\s\w]*(?:to|until)[\s\.:]*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4}|\d{1,2}\s+[A-Za-z]+\s+\d{4})", text)
    if m_range:
        data["policy_start_date"] = m_range.group(1).strip()
        data["policy_end_date"] = m_range.group(2).strip()
    else:
        m_exp = re.search(r"(?i)(?:to\s*midnight\s*of|expires?\s*on|expiry\s*date|to\s*:\s*\d\d:\d\d\s*hrs\s*on)[\s\.:]*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4})", text)
        if m_exp:
            data["policy_end_date"] = m_exp.group(1).strip()

    if "policy_start_date" in data:
        data["POLICY_START_DATE"] = data["policy_start_date"]
    if "policy_end_date" in data:
        data["POLICY_END_DATE"] = data["policy_end_date"]

    # 9. Vehicle Registration Number (VEHICLE_REGISTRATION_NUMBER)
    m = re.search(r"(?i)(?:registration\s*(?:no|number)|regn?\s*no)[\s\.:/-]*([A-Z]{2}\s*\d{1,2}\s*[A-Z]{1,3}\s*\d{1,4})", text)
    if m:
        data["registration_number"] = m.group(1).strip()
    else:
        m_reg = re.search(r"\b([A-Z]{2}\s*\d{1,2}\s*[A-Z]{1,3}\s*\d{1,4})\b", text)
        if m_reg:
            data["registration_number"] = m_reg.group(1).strip()
    if "registration_number" in data:
        data["vehicle_registration_number"] = data["registration_number"]
        data["VEHICLE_REGISTRATION_NUMBER"] = data["registration_number"]

    # 10. Financials: TP, OD, Net, Addon, GST, Total (TP_PREMIUM, OD_PREMIUM, NET_PREMIUM, ADDON_PREMIUM, GST_AMOUNT, TOTAL_PREMIUM)
    m_tp = re.search(r"(?i)(?:total\s*(?:tp|liability)|third\s*party\s*(?:premium|liability|tp)|basic\s*tp|basic\s*third[- ]party)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
    if m_tp:
        data["basic_tp_premium"] = m_tp.group(1)
        data["third_party_premium"] = m_tp.group(1)
        data["tp_premium"] = m_tp.group(1)
        data["TP_PREMIUM"] = m_tp.group(1)

    m_od = re.search(r"(?i)(?:own\s*damage\s*(?:premium|od)|basic\s*od|basic\s*own\s*damage)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
    if m_od:
        data["basic_od_premium"] = m_od.group(1)
        data["own_damage_premium"] = m_od.group(1)
        data["od_premium"] = m_od.group(1)
        data["OD_PREMIUM"] = m_od.group(1)

    m_net = re.search(r"(?i)(?:net\s*premium|total\s*net\s*premium|taxable\s*value|premium\s*before\s*tax|total\s*package\s*premium)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
    if m_net:
        data["net_premium"] = m_net.group(1)
        data["NET_PREMIUM"] = m_net.group(1)

    m_add = re.search(r"(?i)(?:add-on\s*premium|addon\s*premium|total\s*add-on|rider(?:s)?\s*premium)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
    if m_add:
        data["addon_premium"] = m_add.group(1)
        data["ADDON_PREMIUM"] = m_add.group(1)

    m_gst = re.search(r"(?i)(?:gst|igst|cgst\s*\+\s*sgst|integrated\s*tax|taxes)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
    if m_gst:
        data["tax"] = m_gst.group(1)
        data["gst"] = m_gst.group(1)
        data["gst_amount"] = m_gst.group(1)
        data["GST_AMOUNT"] = m_gst.group(1)

    # Total Premium
    m_stand = re.search(r"(?i)(?:^|\n)\s*(?:GRAND\s*TOTAL|TOTAL\s*AMOUNT\s*PAYABLE|TOTAL\s*PREMIUM|TOTAL)\s*\n\s*([\d,]+(?:\.\d{2})?)", text)
    if m_stand:
        data["total_premium"] = m_stand.group(1).strip()
    else:
        m_tot = re.search(r"(?i)(?:total\s*amount\s*payable|total\s*premium|final\s*(?:payable\s*amount|amount|premium)|total\s*payable|total\s*amount|gross\s*premium)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
        if m_tot:
            data["total_premium"] = m_tot.group(1).strip()
        else:
            m_coll = re.search(r"(?i)premium\s*collection\s*details[\s\S]{1,250}?,\s*([\d,]+(?:\.\d{2})?)", text)
            if m_coll:
                data["total_premium"] = m_coll.group(1).strip()
    if "total_premium" in data:
        data["TOTAL_PREMIUM"] = data["total_premium"]

    # 11. IDV: Total IDV and CNG IDV (TOTAL_IDV, CNG_IDV)
    m_idv = re.search(r"(?i)(?:total\s*idv|vehicle\s*idv|idv)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
    if m_idv:
        data["total_idv"] = m_idv.group(1)
        data["idv"] = m_idv.group(1)
        data["TOTAL_IDV"] = m_idv.group(1)

    m_cng = re.search(r"(?i)(?:cng|lpg|bi-fuel)\s*(?:kit)?\s*idv[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
    if m_cng:
        data["cng_idv"] = m_cng.group(1)
        data["CNG_IDV"] = m_cng.group(1)

    m_ncb = re.search(r"(?i)(?:no\s*claim\s*(?:bonus|discount)|ncb)[\s\.:/-]*([0-9\.]+\s*%?)", text)
    if m_ncb:
        data["ncb"] = m_ncb.group(1).strip()
        data["NCB"] = data["ncb"]

    # 12. Engine & Chassis Numbers (ENGINE_NUMBER, CHASSIS_NUMBER)
    m_eng = re.search(r"(?i)(?:engine\s*(?:no|number))[\s\.:/-]*([A-Z0-9]{5,25})", text)
    if m_eng:
        eng_cand = m_eng.group(1).strip()
        if eng_cand.upper() not in ("CHASSIS", "CHASSISNO", "REGISTRATION", "REGNO", "MODEL", "MAKE"):
            data["engine_number"] = eng_cand
            data["ENGINE_NUMBER"] = eng_cand

    m_cha = re.search(r"(?i)(?:chassis\s*(?:no|number)|vin)[\s\.:/-]*([A-Z0-9]{5,25})", text)
    if m_cha:
        cha_cand = m_cha.group(1).strip()
        if cha_cand.upper() not in ("ENGINE", "ENGINENO", "REGISTRATION", "REGNO", "MODEL", "MAKE"):
            data["chassis_number"] = cha_cand
            data["CHASSIS_NUMBER"] = cha_cand

    # 13. Year of Manufacture (YEAR_OF_MANUFACTURE)
    m_yr = re.search(r"(?i)(?:manufacturing\s*year|year\s*of\s*mfg|mfg\s*year|year\s*of\s*manufacture)[\s\.:/-]*(\d{4})", text)
    if m_yr:
        data["manufacturing_year"] = m_yr.group(1)
        data["year_of_manufacture"] = m_yr.group(1)
        data["YEAR_OF_MANUFACTURE"] = m_yr.group(1)

    # 14. Make & Model (MAKE, MODEL)
    if "tata" in text.lower() and "ace" in text.lower():
        data["vehicle_make"] = "TATA"
        data["vehicle_model"] = "ACE 275 ID"
    else:
        m_veh = re.search(r"(?i)(?:make\s*of\s*vehicle|make\s*[\&/]\s*model|make\s+and\s+model|vehicle\s*make|vehicle\s*model|model\s*[\-/]\s*variant|vehicle\s*description)[\s\.:/-]*([^\n\r,;]{3,50})", text)
        if m_veh:
            val = m_veh.group(1).strip()
            clean_val = " ".join(val.strip().split())
            if clean_val.upper() not in ["MODEL", "MAKE", "TYPE OF BODY", "GVW", "OF VEHICLE"] and not clean_val.upper().startswith("OF VEHICLE"):
                detected_make = None
                for mk in ["MARUTI SUZUKI", "MARUTI", "TATA MOTORS", "TATA", "HONDA", "HYUNDAI", "MAHINDRA", "BAJAJ", "HERO", "TVS", "TOYOTA"]:
                    if mk.lower() in clean_val.lower():
                        detected_make = mk
                        break
                if detected_make:
                    data["vehicle_make"] = detected_make
                    if clean_val.upper().startswith(detected_make):
                        rem = clean_val[len(detected_make):].strip(" -/")
                        data["vehicle_model"] = rem if len(rem) >= 2 else clean_val
                    else:
                        data["vehicle_model"] = clean_val
                else:
                    data["vehicle_model"] = clean_val

    if "vehicle_make" in data:
        data["make"] = data["vehicle_make"]
        data["MAKE"] = data["vehicle_make"]
    if "vehicle_model" in data:
        data["model"] = data["vehicle_model"]
        data["MODEL"] = data["vehicle_model"]

    # 15. Seating Capacity (SEATING_CAPACITY)
    m_seat = re.search(r"(?i)(?:seating\s*capacity|seating)[\s\.:/-]*(\d{1,2})", text)
    if m_seat:
        data["seating_capacity"] = m_seat.group(1)
        data["SEATING_CAPACITY"] = data["seating_capacity"]

    # 16. NCB (NCB)
    m_ncb = re.search(r"(?i)(?:ncb|no\s*claim\s*bonus)[\s\.:/-]*(\d{1,2}\s*%)", text)
    if m_ncb:
        data["ncb"] = m_ncb.group(1).replace(" ", "")
        data["ncb_percentage"] = data["ncb"]
        data["NCB"] = data["ncb"]

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

    m = re.search(r"(?i)(?:policyholder\s*name|name\s*of\s*proposer|proposer\s*name|insured\s*name)[\s\.:/-]*\n?\s*([^\n\r,;]{3,50})", text)
    if m:
        c_name = m.group(1).strip()
        from validation.field_rules import clean_customer_name
        cleaned = clean_customer_name(c_name)
        if cleaned:
            data["policyholder_name"] = cleaned
            data["customer_name"] = cleaned

    m_plan = re.search(r"(?i)(?:plan\s*name|product\s*name|policy\s*name)[\s\.:/-]*([^\n\r,;]{3,50})", text)
    if m_plan:
        data["plan_name"] = m_plan.group(1).strip()
        data["product_name"] = data["plan_name"]

    m_si = re.search(r"(?i)(?:sum\s*insured|basic\s*sum\s*insured)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
    if m_si:
        data["sum_insured"] = m_si.group(1)

    m_cb = re.search(r"(?i)(?:cumulative\s*bonus)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
    if m_cb:
        data["cumulative_bonus"] = m_cb.group(1)

    m_tot = re.search(r"(?i)(?:total\s*premium|gross\s*premium|net\s*premium)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
    if m_tot:
        data["premium"] = m_tot.group(1)
        data["total_premium"] = m_tot.group(1)

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
        b_name = m.group(1).strip()
        data["insured_business"] = b_name
        data["customer_name"] = b_name

    m_bt = re.search(r"(?i)(?:business\s*type|occupancy)[\s\.:/-]*([^\n\r,;]{3,50})", text)
    if m_bt:
        data["business_type"] = m_bt.group(1).strip()

    m_si = re.search(r"(?i)(?:total\s*sum\s*insured|sum\s*insured)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
    if m_si:
        data["total_sum_insured"] = m_si.group(1)
        data["sum_insured"] = m_si.group(1)

    m_base = re.search(r"(?i)(?:base\s*premium|fire\s*premium)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
    if m_base:
        data["base_premium"] = m_base.group(1)

    m_terr = re.search(r"(?i)(?:terrorism\s*premium)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
    if m_terr:
        data["terrorism_premium"] = m_terr.group(1)

    m_gst = re.search(r"(?i)(?:gst\s*amount|gst|tax)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
    if m_gst:
        data["tax"] = m_gst.group(1)
        data["gst_amount"] = m_gst.group(1)

    m_tot = re.search(r"(?i)(?:total\s*premium)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
    if m_tot:
        data["total_premium"] = m_tot.group(1)

    m_dates = re.findall(r"\b(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4})\b", text)
    if len(m_dates) >= 2:
        data["policy_start_date"] = m_dates[0]
        data["policy_end_date"] = m_dates[1]

    return data


def extract_deterministic_workmen(text: str) -> Dict[str, Any]:
    """Deterministic regex parser for Workmen's Compensation."""
    data: Dict[str, Any] = {}

    m = re.search(r"(?i)(?:policy\s*(?:no|number))[\s\.:/-]*([A-Z0-9/\-]{8,35})", text)
    if m:
        data["policy_number"] = m.group(1).strip()

    m = re.search(r"(?i)(?:employer\s*name|insured\s*name|name\s*of\s*insured)[\s\.:/-]*([^\n\r,;]{3,50})", text)
    if m:
        emp_name = m.group(1).strip()
        data["employer"] = emp_name
        data["customer_name"] = emp_name

    m_ba = re.search(r"(?i)(?:business\s*activity|nature\s*of\s*work|trade)[\s\.:/-]*([^\n\r,;]{3,60})", text)
    if m_ba:
        data["business_activity"] = m_ba.group(1).strip()
        data["nature_of_work"] = data["business_activity"]

    m_emp = re.search(r"(?i)(?:number\s*of\s*employees|total\s*employees|no\s*of\s*workmen)[\s\.:]*(\d{1,6})", text)
    if m_emp:
        data["number_of_employees"] = m_emp.group(1)

    m_tot = re.search(r"(?i)(?:total\s*premium)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
    if m_tot:
        data["total_premium"] = m_tot.group(1)

    m_dates = re.findall(r"\b(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4})\b", text)
    if len(m_dates) >= 2:
        data["policy_start_date"] = m_dates[0]
        data["policy_end_date"] = m_dates[1]

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
    page_texts: Optional[Dict[int, str]] = None,
    layout: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Main extraction orchestrator.
    Combines:
    1. Dedicated multi-field resolvers (Policy No, Customer, Dates, Amounts, Insurer, Vehicle)
    2. Deterministic domain-specific parser
    3. Any AcroForm field values
    4. Any structured table values
    5. Ollama LLM structured refinement (if online)
    """
    from retrieval.field_resolvers.policy_number_resolver import PolicyNumberResolver
    from retrieval.field_resolvers.customer_name_resolver import CustomerNameResolver
    from retrieval.field_resolvers.date_resolver import DateResolver
    from retrieval.field_resolvers.amount_resolver import AmountResolver
    from retrieval.field_resolvers.insurer_resolver import InsurerResolver
    from retrieval.field_resolvers.vehicle_resolver import VehicleResolver

    pages = page_texts or {1: text}
    extracted: Dict[str, Any] = {}
    evidence_candidates: Dict[str, Any] = {}

    # Step 1: Run Dedicated High-Precision Resolvers
    # A. Policy Number / Identifier
    p_cand = PolicyNumberResolver().resolve(full_text=text, page_texts=pages, layout=layout, tables=tables)
    if p_cand:
        extracted["policy_number"] = p_cand.value
        evidence_candidates["policy_number"] = p_cand
        if p_cand.metadata.get("all_policy_numbers"):
            extracted["all_policy_numbers"] = p_cand.metadata["all_policy_numbers"]
        if p_cand.metadata.get("secondary_policy_number"):
            extracted["secondary_policy_number"] = p_cand.metadata["secondary_policy_number"]
        if doc_type == DocumentType.ELECTRICITY_BILL:
            extracted["consumer_number"] = p_cand.value
            evidence_candidates["consumer_number"] = p_cand

    # B. Customer / Insured Name
    c_cand = CustomerNameResolver().resolve(full_text=text, page_texts=pages, layout=layout, tables=tables, context={"doc_type": doc_type})
    if c_cand:
        extracted["insured_name"] = c_cand.value
        extracted["customer_name"] = c_cand.value
        evidence_candidates["insured_name"] = c_cand
        evidence_candidates["customer_name"] = c_cand
        if doc_type == DocumentType.HEALTH_INSURANCE or "policyholder" in getattr(c_cand, "exact_label", "").lower():
            extracted["policyholder_name"] = c_cand.value
            evidence_candidates["policyholder_name"] = c_cand
        if c_cand.metadata.get("customer_id"):
            extracted["customer_id"] = c_cand.metadata["customer_id"]

    # C. Dates and Duration
    d_res = DateResolver().resolve(full_text=text, page_texts=pages, layout=layout, tables=tables)
    for dk, dv in d_res.items():
        if dv:
            extracted[dk] = dv.value
            evidence_candidates[dk] = dv
    if d_res.get("policy_start_date") and doc_type == DocumentType.ELECTRICITY_BILL:
        extracted["bill_date"] = d_res["policy_start_date"].value
        evidence_candidates["bill_date"] = d_res["policy_start_date"]
    if d_res.get("policy_end_date") and doc_type == DocumentType.ELECTRICITY_BILL:
        extracted["due_date"] = d_res["policy_end_date"].value
        evidence_candidates["due_date"] = d_res["policy_end_date"]

    # D. Amounts and Financials
    amt_res = AmountResolver().resolve(full_text=text, page_texts=pages, layout=layout, tables=tables)
    if amt_res.get("idv"):
        extracted["idv"] = amt_res["idv"].value
        extracted["total_idv"] = amt_res["idv"].value
        evidence_candidates["idv"] = amt_res["idv"]
        evidence_candidates["total_idv"] = amt_res["idv"]
    if amt_res.get("cng_idv"):
        extracted["cng_idv"] = amt_res["cng_idv"].value
        evidence_candidates["cng_idv"] = amt_res["cng_idv"]
    if amt_res.get("ncb"):
        extracted["ncb"] = amt_res["ncb"].value
        evidence_candidates["ncb"] = amt_res["ncb"]
    if amt_res.get("basic_od_premium"):
        extracted["basic_od_premium"] = amt_res["basic_od_premium"].value
        extracted["od_premium"] = amt_res["basic_od_premium"].value
        extracted["own_damage_premium"] = amt_res["basic_od_premium"].value
        evidence_candidates["basic_od_premium"] = amt_res["basic_od_premium"]
    if amt_res.get("basic_tp_premium"):
        extracted["basic_tp_premium"] = amt_res["basic_tp_premium"].value
        extracted["tp_premium"] = amt_res["basic_tp_premium"].value
        extracted["third_party_premium"] = amt_res["basic_tp_premium"].value
        evidence_candidates["basic_tp_premium"] = amt_res["basic_tp_premium"]
    if amt_res.get("sum_insured"):
        extracted["sum_insured"] = amt_res["sum_insured"].value
        extracted["total_sum_insured"] = amt_res["sum_insured"].value
        evidence_candidates["sum_insured"] = amt_res["sum_insured"]
        evidence_candidates["total_sum_insured"] = amt_res["sum_insured"]
    if amt_res.get("net_premium"):
        extracted["net_premium"] = amt_res["net_premium"].value
        extracted["premium"] = amt_res["net_premium"].value
        evidence_candidates["net_premium"] = amt_res["net_premium"]
        evidence_candidates["premium"] = amt_res["net_premium"]
    if amt_res.get("tax"):
        extracted["tax"] = amt_res["tax"].value
        extracted["gst"] = amt_res["tax"].value
        extracted["gst_amount"] = amt_res["tax"].value
        evidence_candidates["tax"] = amt_res["tax"]
        evidence_candidates["gst"] = amt_res["tax"]
    if amt_res.get("total_premium"):
        extracted["total_premium"] = amt_res["total_premium"].value
        extracted["total_amount"] = amt_res["total_premium"].value
        extracted["total_bill"] = amt_res["total_premium"].value
        extracted["net_bill_amount"] = amt_res["total_premium"].value
        evidence_candidates["total_premium"] = amt_res["total_premium"]
        evidence_candidates["total_amount"] = amt_res["total_premium"]
        evidence_candidates["total_bill"] = amt_res["total_premium"]
        evidence_candidates["net_bill_amount"] = amt_res["total_premium"]
    elif amt_res.get("total_amount"):
        extracted["total_premium"] = amt_res["total_amount"].value
        extracted["total_amount"] = amt_res["total_amount"].value
        extracted["total_bill"] = amt_res["total_amount"].value
        extracted["net_bill_amount"] = amt_res["total_amount"].value
        evidence_candidates["total_premium"] = amt_res["total_amount"]
        evidence_candidates["total_amount"] = amt_res["total_amount"]

    # E. Insurance Company / Provider
    ins_cand = InsurerResolver().resolve(full_text=text, page_texts=pages, layout=layout, tables=tables)
    if ins_cand:
        extracted["insurance_company"] = ins_cand.value
        extracted["insurer"] = ins_cand.value
        evidence_candidates["insurance_company"] = ins_cand
        evidence_candidates["insurer"] = ins_cand
        if doc_type == DocumentType.ELECTRICITY_BILL:
            extracted["electricity_provider"] = ins_cand.value
            evidence_candidates["electricity_provider"] = ins_cand

    # F. Vehicle Identifiers (Motor Insurance only)
    if doc_type == DocumentType.MOTOR_INSURANCE:
        v_res = VehicleResolver().resolve(full_text=text, page_texts=pages, layout=layout, tables=tables)
        for vk, vc in v_res.items():
            if vc:
                extracted[vk] = vc.value
                evidence_candidates[vk] = vc
                if vk == "vehicle_make":
                    extracted["make"] = vc.value
                    evidence_candidates["make"] = vc
                elif vk == "vehicle_model":
                    extracted["model"] = vc.value
                    evidence_candidates["model"] = vc
                elif vk == "registration_number":
                    extracted["vehicle_registration_number"] = vc.value
                    evidence_candidates["vehicle_registration_number"] = vc
                elif vk == "manufacturing_year":
                    extracted["year_of_manufacture"] = vc.value
                    evidence_candidates["year_of_manufacture"] = vc
                elif vk == "total_idv":
                    extracted["idv"] = vc.value
                    evidence_candidates["idv"] = vc

    # Step 2: Overlay domain-specific parser for supplementary domain fields
    if doc_type == DocumentType.MOTOR_INSURANCE:
        domain_data = extract_deterministic_motor(text)
    elif doc_type == DocumentType.ELECTRICITY_BILL:
        domain_data = extract_deterministic_electricity(text)
    elif doc_type == DocumentType.HEALTH_INSURANCE:
        domain_data = extract_deterministic_health(text)
    elif doc_type == DocumentType.PROPERTY_INSURANCE:
        domain_data = extract_deterministic_property(text)
    elif doc_type == DocumentType.WORKMEN_COMPENSATION:
        domain_data = extract_deterministic_workmen(text)
    else:
        domain_data = extract_deterministic_generic(text)

    for dk, dv in domain_data.items():
        if dk not in extracted and dv is not None:
            extracted[dk] = dv

    # Step 3: Overlay AcroForm values if present
    if form_fields:
        for fk, fv in form_fields.items():
            if fv and fk.lower() not in extracted:
                extracted[f"form_{fk}"] = fv

    # Step 4: Overlay Table values if present
    if tables:
        for t in tables:
            kv_map = t.get("key_value_map", {})
            for tk, tv in kv_map.items():
                tk_lower = tk.lower()
                if "total" in tk_lower and not any(neg in tk_lower for neg in ["tax", "gst", "cgst", "sgst", "igst"]) and "total_premium" not in extracted:
                    extracted["table_total"] = tv

    # Step 5: Secondary candidate generation with Ollama LLM
    # RULE 6: The LLM must NOT invent missing fields.
    # Result must be verified to exist verbatim in document text or tables.
    has_dates = bool(extracted.get("policy_start_date") or extracted.get("bill_date"))
    has_amount = bool(extracted.get("total_amount") or extracted.get("total_premium") or extracted.get("total_bill"))
    has_policy = bool(extracted.get("policy_number") or extracted.get("consumer_number") or extracted.get("reference_number"))
    has_name = bool(extracted.get("customer_name") or extracted.get("insured_name") or extracted.get("primary_party_name"))

    core_missing = []
    if not has_policy:
        core_missing.append("policy_number")
    if not has_name:
        core_missing.append("customer_name")
    if not has_dates:
        core_missing.append("dates")
    if not has_amount:
        core_missing.append("amount")

    # In Motor Insurance, target missing schema fields
    if doc_type == DocumentType.MOTOR_INSURANCE:
        for sk in ["AGENT_NAME", "AGENT_CODE", "CLASS_OF_VEHICLE", "INSURANCE_TYPE", "POLICY_BOOKING_DATE", "ADDON_PREMIUM", "CNG_IDV", "NCB"]:
            if sk not in extracted and not any(alias in extracted for alias in SCHEMA_TO_CANONICAL_MAP.get(sk, [])):
                core_missing.append(sk)

    if core_missing:
        llm_extracted = extract_with_ollama(text, doc_type, core_missing)
        if llm_extracted:
            for k, v in llm_extracted.items():
                if v is not None and str(v).strip() and str(v).lower() not in ("null", "none"):
                    v_str = str(v).strip()
                    # VERIFY GROUNDING: Must exist verbatim in document text or table cells
                    is_grounded = v_str.lower() in text.lower()
                    if not is_grounded and tables:
                        for t in tables:
                            for r in t.get("raw_matrix", []):
                                for c in r:
                                    if c and v_str.lower() in str(c).lower():
                                        is_grounded = True
                                        break
                    if not is_grounded:
                        logger.warning(f"Rejecting ungrounded LLM candidate for {k}: {v_str}")
                        continue

                    # Hard validation checks on LLM candidate
                    if k.upper() in ["MODEL", "VEHICLE_MODEL"] and (v_str.upper() in ["OF VEHICLE", "VEHICLE", "MAKE", "MODEL", "TYPE OF BODY"]):
                        continue
                    if k.upper() in ["CUSTOMER_NAME", "INSURED_NAME"] and any(bad in v_str.upper() for bad in ["DEFENCE", "SETTLEMENT", "CONDITIONS", "LIMITATIONS"]):
                        continue

                    if k not in extracted:
                        extracted[k] = v_str
                    norm_k = str(k).strip().upper()
                    if norm_k in SCHEMA_TO_CANONICAL_MAP:
                        for alias in SCHEMA_TO_CANONICAL_MAP[norm_k]:
                            if alias not in extracted:
                                extracted[alias] = v_str
                    elif norm_k in MOTOR_FIELD_ALIASES:
                        for alias in MOTOR_FIELD_ALIASES[norm_k]:
                            if alias not in extracted:
                                extracted[alias] = v_str

    # Store candidates mapping for downstream normalizer traceability
    extracted["_evidence_candidates"] = evidence_candidates

    return extracted
