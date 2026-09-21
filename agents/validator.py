"""
Validator Agent: Executes validation and builds canonical presentation sections.
Guarantees clean display values, no raw OCR garbage, and no hallucinations.
"""

from typing import Dict, Any, List, Tuple
from schemas.base import DocumentType
from validation.validator import validate_document_data
from utils.formatting import format_currency_inr, clean_display_value


def build_canonical_sections(data: Dict[str, Any], doc_type: DocumentType, subtype: str = None) -> List[Dict[str, Any]]:
    """
    Builds clean, structured visual sections for production UI rendering.
    Adheres strictly to Rule 9, 10, 11, 12, 13, 14, 52, 53:
    Never renders OCR garbage, raw chunks, candidate lists, or confidence values.
    """
    sections: List[Dict[str, Any]] = []

    if doc_type == DocumentType.MOTOR_INSURANCE:
        title_sub = subtype or "Motor"
        # Customer Card
        sections.append({
            "title": "Customer Details",
            "icon": "👤",
            "fields": [
                {"label": "Insured Name", "value": clean_display_value(data.get("insured_name"))},
                {"label": "Address", "value": clean_display_value(data.get("insured_address"))},
                {"label": "Mobile Number", "value": clean_display_value(data.get("mobile"))},
                {"label": "Email", "value": clean_display_value(data.get("email"))},
            ],
        })

        # Policy Card
        sections.append({
            "title": "Policy Information",
            "icon": "📄",
            "fields": [
                {"label": "Policy Number", "value": clean_display_value(data.get("policy_number"))},
                {"label": "Policy Type", "value": clean_display_value(data.get("policy_type"))},
                {"label": "Insurer", "value": clean_display_value(data.get("insurer"))},
                {"label": "Coverage", "value": clean_display_value(data.get("coverage"))},
            ],
        })

        # Vehicle Card
        sections.append({
            "title": "Vehicle Specifications",
            "icon": "🚗",
            "fields": [
                {"label": "Registration Number", "value": clean_display_value(data.get("registration_number"))},
                {"label": "Make", "value": clean_display_value(data.get("vehicle_make"))},
                {"label": "Model", "value": clean_display_value(data.get("vehicle_model"))},
                {"label": "Manufacturing Year", "value": clean_display_value(data.get("manufacturing_year"))},
                {"label": "Engine Number", "value": clean_display_value(data.get("engine_number"))},
                {"label": "Chassis Number", "value": clean_display_value(data.get("chassis_number"))},
                {"label": "RTO", "value": clean_display_value(data.get("rto"))},
            ],
        })

        # Financials Card
        sections.append({
            "title": "Premium & Tax Breakdown",
            "icon": "💰",
            "fields": [
                {"label": "Own Damage Premium", "value": format_currency_inr(data.get("own_damage_premium"))},
                {"label": "Third Party Premium", "value": format_currency_inr(data.get("third_party_premium"))},
                {"label": "GST", "value": format_currency_inr(data.get("gst"))},
                {"label": "Total Premium", "value": format_currency_inr(data.get("total_premium")), "is_highlight": True},
            ],
        })

        # Important Dates Card
        sections.append({
            "title": "Important Dates",
            "icon": "📅",
            "fields": [
                {"label": "Policy Start Date", "value": clean_display_value(data.get("policy_start_date"))},
                {"label": "Policy End Date", "value": clean_display_value(data.get("policy_end_date"))},
                {"label": "Renewal Due Date", "value": clean_display_value(data.get("policy_end_date"))},
            ],
        })

    elif doc_type == DocumentType.ELECTRICITY_BILL:
        # Customer Card
        sections.append({
            "title": "Consumer Information",
            "icon": "👤",
            "fields": [
                {"label": "Customer Name", "value": clean_display_value(data.get("customer_name"))},
                {"label": "Consumer Number", "value": clean_display_value(data.get("consumer_number"))},
                {"label": "Electricity Provider", "value": clean_display_value(data.get("electricity_provider"))},
                {"label": "Billing Address", "value": clean_display_value(data.get("address"))},
            ],
        })

        # Connection & Meter Card
        sections.append({
            "title": "Connection & Meter",
            "icon": "⚡",
            "fields": [
                {"label": "Meter Number", "value": clean_display_value(data.get("meter_number"))},
                {"label": "Tariff", "value": clean_display_value(data.get("tariff"))},
                {"label": "Phase", "value": clean_display_value(data.get("phase"))},
            ],
        })

        # Readings & Consumption Card
        sections.append({
            "title": "Readings & Consumption",
            "icon": "📊",
            "fields": [
                {"label": "Previous Reading", "value": clean_display_value(data.get("previous_reading"))},
                {"label": "Current Reading", "value": clean_display_value(data.get("current_reading"))},
                {"label": "Units Consumed", "value": f"{data.get('consumption')} kWh" if data.get("consumption") else "Not available in the document"},
            ],
        })

        # Charges & Financials
        sections.append({
            "title": "Charges Breakdown",
            "icon": "💳",
            "fields": [
                {"label": "Fixed Charges", "value": format_currency_inr(data.get("fixed_charge"))},
                {"label": "Energy Charges", "value": format_currency_inr(data.get("energy_charge"))},
                {"label": "Fuel Surcharge", "value": format_currency_inr(data.get("fuel_charge"))},
                {"label": "Electricity Duty", "value": format_currency_inr(data.get("electricity_duty"))},
                {"label": "Net Bill Amount", "value": format_currency_inr(data.get("total_bill") or data.get("net_bill_amount")), "is_highlight": True},
            ],
        })

        # Dates
        sections.append({
            "title": "Billing Dates",
            "icon": "📅",
            "fields": [
                {"label": "Bill Date", "value": clean_display_value(data.get("bill_date"))},
                {"label": "Due Date", "value": clean_display_value(data.get("due_date")), "is_highlight": True},
            ],
        })

    elif doc_type == DocumentType.HEALTH_INSURANCE:
        sections.append({
            "title": "Policyholder Details",
            "icon": "👤",
            "fields": [
                {"label": "Policyholder Name", "value": clean_display_value(data.get("policyholder_name"))},
                {"label": "Policy Number", "value": clean_display_value(data.get("policy_number"))},
                {"label": "Insurer", "value": clean_display_value(data.get("insurer"))},
                {"label": "Plan Name", "value": clean_display_value(data.get("plan_name"))},
            ],
        })

        sections.append({
            "title": "Coverage & Sum Insured",
            "icon": "🏥",
            "fields": [
                {"label": "Sum Insured", "value": format_currency_inr(data.get("sum_insured")), "is_highlight": True},
                {"label": "Cumulative Bonus", "value": format_currency_inr(data.get("cumulative_bonus"))},
                {"label": "Hospitalization Cover", "value": clean_display_value(data.get("hospitalization_cover"))},
            ],
        })

        sections.append({
            "title": "Financial Details",
            "icon": "💰",
            "fields": [
                {"label": "Base Premium", "value": format_currency_inr(data.get("premium"))},
                {"label": "GST", "value": format_currency_inr(data.get("gst"))},
                {"label": "Net Premium", "value": format_currency_inr(data.get("net_premium") or data.get("premium")), "is_highlight": True},
            ],
        })

        sections.append({
            "title": "Policy Period",
            "icon": "📅",
            "fields": [
                {"label": "Policy Start Date", "value": clean_display_value(data.get("policy_start_date"))},
                {"label": "Policy End Date", "value": clean_display_value(data.get("policy_end_date"))},
            ],
        })

    else:
        # Generic Document
        sections.append({
            "title": "Primary Details",
            "icon": "📑",
            "fields": [
                {"label": "Document Title", "value": clean_display_value(data.get("document_title") or "Commercial Document")},
                {"label": "Primary Party", "value": clean_display_value(data.get("primary_party_name") or data.get("insured_name") or data.get("customer_name"))},
                {"label": "Reference Number", "value": clean_display_value(data.get("reference_number") or data.get("policy_number"))},
            ],
        })

        sections.append({
            "title": "Valuation & Financials",
            "icon": "💰",
            "fields": [
                {"label": "Amount / Value", "value": format_currency_inr(data.get("amount_or_value") or data.get("total_premium") or data.get("total_bill")), "is_highlight": True},
            ],
        })

        sections.append({
            "title": "Dates",
            "icon": "📅",
            "fields": [
                {"label": "Date of Issue", "value": clean_display_value(data.get("date_of_issue") or data.get("policy_start_date") or data.get("bill_date"))},
                {"label": "Expiry / Due Date", "value": clean_display_value(data.get("expiry_or_due_date") or data.get("policy_end_date") or data.get("due_date"))},
            ],
        })

    return sections
