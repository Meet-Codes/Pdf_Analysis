"""
Field Normalizer: Coordinates field-level deterministic normalization.
Transforms raw extracted strings into canonical values adhering to schema specifications.
"""

from typing import Dict, Any
from schemas.base import DocumentType
from validation.field_rules import (
    clean_customer_name,
    clean_registration_number,
    clean_policy_number,
    clean_phone_number,
    clean_vehicle_class,
    clean_insurance_type,
    clean_ncb,
    clean_seating_capacity,
    clean_year,
    clean_engine_number,
    clean_chassis_number,
)
from validation.dates import parse_and_normalize_date
from validation.financial import parse_currency_amount


def normalize_fields(data: Dict[str, Any], doc_type: DocumentType) -> Dict[str, Any]:
    """
    Normalizes a dictionary of extracted raw fields into canonical representation.
    """
    normalized = {}

    # Target keys for special handling
    name_keys = {"insured_name", "customer_name", "policyholder_name", "employer", "primary_party_name", "agent_name"}
    policy_keys = {"policy_number", "secondary_policy_number", "proposal_number", "bill_number", "consumer_number", "meter_number", "reference_number", "agent_code"}
    date_keys = {"policy_start_date", "policy_end_date", "registration_date", "bill_date", "due_date", "date_of_issue", "expiry_or_due_date", "policy_booking_date", "booking_date"}
    financial_keys = {
        "idv", "total_idv", "cng_idv", "own_damage_premium", "od_premium", "third_party_premium", "tp_premium", "gst", "gst_amount", "total_premium", "premium",
        "sum_insured", "net_premium", "addon_premium", "fixed_charge", "energy_charge", "fuel_charge", "electricity_duty",
        "meter_charge", "other_charges", "arrears", "adjustments", "interest", "total_bill", "net_bill_amount",
        "previous_bill_amount", "amount_or_value", "base_premium", "terrorism_premium", "total_sum_insured"
    }

    for k, v in data.items():
        if v is None:
            normalized[k] = None
            continue

        if k in name_keys:
            normalized[k] = clean_customer_name(str(v))
        elif k in ("registration_number", "vehicle_registration_number"):
            normalized[k] = clean_registration_number(str(v))
        elif k in policy_keys:
            normalized[k] = clean_policy_number(str(v))
        elif k in ("mobile", "phone", "customer_mobile"):
            normalized[k] = clean_phone_number(str(v))
        elif k in date_keys:
            normalized[k] = parse_and_normalize_date(str(v))
        elif k in financial_keys:
            normalized[k] = parse_currency_amount(v)
        elif k in ("ncb", "ncb_percentage"):
            normalized[k] = clean_ncb(str(v))
        elif k == "seating_capacity":
            normalized[k] = clean_seating_capacity(str(v))
        elif k in ("class_of_vehicle", "vehicle_type"):
            normalized[k] = clean_vehicle_class(str(v))
        elif k in ("insurance_type", "policy_type"):
            normalized[k] = clean_insurance_type(str(v))
        elif k in ("year_of_manufacture", "manufacturing_year"):
            normalized[k] = clean_year(str(v))
        elif k in ("engine_number", "engine_no"):
            normalized[k] = clean_engine_number(str(v))
        elif k in ("chassis_number", "chassis_no", "vin"):
            normalized[k] = clean_chassis_number(str(v))
        elif k == "all_policy_numbers" and isinstance(v, list):
            normalized[k] = [clean_policy_number(p) for p in v if clean_policy_number(p)]
        elif isinstance(v, str):
            # General string cleanup
            cleaned = v.strip()
            # Collapse internal extra spaces
            cleaned = " ".join(cleaned.split())
            normalized[k] = cleaned if cleaned else None
        elif isinstance(v, list):
            # List of strings or nested objects
            norm_list = []
            for item in v:
                if isinstance(item, dict):
                    norm_list.append(normalize_fields(item, doc_type))
                elif isinstance(item, str):
                    s = item.strip()
                    if s:
                        norm_list.append(s)
                else:
                    norm_list.append(item)
            normalized[k] = norm_list
        elif isinstance(v, dict):
            normalized[k] = normalize_fields(v, doc_type)
        else:
            normalized[k] = v

    return normalized
