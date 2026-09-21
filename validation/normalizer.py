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
)
from validation.dates import parse_and_normalize_date
from validation.financial import parse_currency_amount


def normalize_fields(data: Dict[str, Any], doc_type: DocumentType) -> Dict[str, Any]:
    """
    Normalizes a dictionary of extracted raw fields into canonical representation.
    """
    normalized = {}

    # Target keys for special handling
    name_keys = {"insured_name", "customer_name", "policyholder_name", "employer", "primary_party_name"}
    policy_keys = {"policy_number", "proposal_number", "bill_number", "consumer_number", "meter_number", "reference_number"}
    date_keys = {"policy_start_date", "policy_end_date", "registration_date", "bill_date", "due_date", "date_of_issue", "expiry_or_due_date"}
    financial_keys = {
        "idv", "own_damage_premium", "third_party_premium", "gst", "total_premium", "premium",
        "sum_insured", "net_premium", "fixed_charge", "energy_charge", "fuel_charge", "electricity_duty",
        "meter_charge", "other_charges", "arrears", "adjustments", "interest", "total_bill", "net_bill_amount",
        "previous_bill_amount", "amount_or_value", "base_premium", "terrorism_premium", "gst_amount", "total_sum_insured"
    }

    for k, v in data.items():
        if v is None:
            normalized[k] = None
            continue

        if k in name_keys:
            normalized[k] = clean_customer_name(str(v))
        elif k == "registration_number":
            normalized[k] = clean_registration_number(str(v))
        elif k in policy_keys:
            normalized[k] = clean_policy_number(str(v))
        elif k in ("mobile", "phone"):
            normalized[k] = clean_phone_number(str(v))
        elif k in date_keys:
            normalized[k] = parse_and_normalize_date(str(v))
        elif k in financial_keys:
            normalized[k] = parse_currency_amount(v)
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
