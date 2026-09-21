"""
Summarizer Agent: Generates concise, business-oriented document summaries.
Strictly derived from validated canonical structured data. Never hallucinates.
"""

from typing import Dict, Any
from schemas.base import DocumentType
from utils.formatting import format_currency_inr, clean_display_value


def generate_document_summary(data: Dict[str, Any], doc_type: DocumentType, subtype: str = None) -> str:
    """
    Generates a clean human-readable executive summary directly from canonical structured data.
    Rule 75: No speculation, no mention of OCR, chunks, or internal mechanics.
    """
    lines = []

    if doc_type == DocumentType.MOTOR_INSURANCE:
        title = f"{subtype or 'Motor'} Insurance Policy".title()
        customer = clean_display_value(data.get("insured_name"))
        policy = clean_display_value(data.get("policy_number"))
        veh_parts = [p for p in [data.get("vehicle_make"), data.get("vehicle_model")] if p]
        vehicle = " ".join(veh_parts) if veh_parts else "Not available in the document"
        reg_no = clean_display_value(data.get("registration_number"))

        start_dt = clean_display_value(data.get("policy_start_date"))
        end_dt = clean_display_value(data.get("policy_end_date"))
        period = f"{start_dt} – {end_dt}" if start_dt != "Not available in the document" else "Not available in the document"

        prem = format_currency_inr(data.get("total_premium"))

        lines.append(f"**Document**: {title}")
        lines.append(f"**Insured Customer**: {customer}")
        lines.append(f"**Policy Number**: {policy}")
        lines.append(f"**Insured Vehicle**: {vehicle} ({reg_no})")
        lines.append(f"**Policy Period**: {period}")
        lines.append(f"**Total Premium**: {prem}")

    elif doc_type == DocumentType.ELECTRICITY_BILL:
        provider = clean_display_value(data.get("electricity_provider") or "Electricity Distribution Utility")
        customer = clean_display_value(data.get("customer_name"))
        consumer_no = clean_display_value(data.get("consumer_number"))
        meter_no = clean_display_value(data.get("meter_number"))
        consumption = f"{data.get('consumption')} kWh" if data.get("consumption") else "Not available in the document"
        due_date = clean_display_value(data.get("due_date"))
        bill_amount = format_currency_inr(data.get("total_bill") or data.get("net_bill_amount"))

        lines.append(f"**Document**: Electricity Bill ({provider})")
        lines.append(f"**Consumer Name**: {customer}")
        lines.append(f"**Consumer Number**: {consumer_no}")
        lines.append(f"**Meter Number**: {meter_no}")
        lines.append(f"**Consumption**: {consumption}")
        lines.append(f"**Due Date**: {due_date}")
        lines.append(f"**Net Bill Amount**: {bill_amount}")

    elif doc_type == DocumentType.HEALTH_INSURANCE:
        customer = clean_display_value(data.get("policyholder_name"))
        policy = clean_display_value(data.get("policy_number"))
        sum_insured = format_currency_inr(data.get("sum_insured"))
        start_dt = clean_display_value(data.get("policy_start_date"))
        end_dt = clean_display_value(data.get("policy_end_date"))
        period = f"{start_dt} – {end_dt}" if start_dt != "Not available in the document" else "Not available in the document"
        prem = format_currency_inr(data.get("net_premium") or data.get("premium"))

        lines.append(f"**Document**: Health Insurance Policy")
        lines.append(f"**Policyholder**: {customer}")
        lines.append(f"**Policy Number**: {policy}")
        lines.append(f"**Sum Insured**: {sum_insured}")
        lines.append(f"**Policy Period**: {period}")
        lines.append(f"**Premium**: {prem}")

    else:
        party = clean_display_value(data.get("primary_party_name") or data.get("insured_name") or data.get("customer_name"))
        ref = clean_display_value(data.get("reference_number") or data.get("policy_number"))
        amt = format_currency_inr(data.get("amount_or_value") or data.get("total_premium") or data.get("total_bill"))

        lines.append(f"**Document**: {doc_type.value.replace('_', ' ').title()}")
        lines.append(f"**Primary Party**: {party}")
        lines.append(f"**Reference / ID**: {ref}")
        lines.append(f"**Valuation / Amount**: {amt}")

    return "\n\n".join(lines)
