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
                {"label": "Insured Name", "value": clean_display_value(data.get("customer_name") or data.get("insured_name"))},
                {"label": "Address", "value": clean_display_value(data.get("insured_address"))},
                {"label": "Mobile Number", "value": clean_display_value(data.get("customer_mobile") or data.get("mobile"))},
                {"label": "Email", "value": clean_display_value(data.get("email"))},
            ],
        })

        # Policy & Intermediary Card
        sections.append({
            "title": "Policy Information",
            "icon": "📄",
            "fields": [
                {"label": "Policy Number", "value": clean_display_value(data.get("policy_number"))},
                {"label": "Insurance Type", "value": clean_display_value(data.get("insurance_type") or data.get("policy_type"))},
                {"label": "Insurer", "value": clean_display_value(data.get("insurance_company") or data.get("company_name") or data.get("insurer"))},
                {"label": "Agent Name", "value": clean_display_value(data.get("agent_name"))},
                {"label": "Agent Code", "value": clean_display_value(data.get("agent_code"))},
                {"label": "Coverage", "value": clean_display_value(data.get("coverage"))},
            ],
        })

        # Vehicle Card
        sections.append({
            "title": "Vehicle Specifications",
            "icon": "🚗",
            "fields": [
                {"label": "Registration Number", "value": clean_display_value(data.get("registration_number") or data.get("vehicle_registration_number"))},
                {"label": "Class of Vehicle", "value": clean_display_value(data.get("class_of_vehicle") or data.get("vehicle_type"))},
                {"label": "Make", "value": clean_display_value(data.get("vehicle_make") or data.get("make"))},
                {"label": "Model", "value": clean_display_value(data.get("vehicle_model") or data.get("model"))},
                {"label": "Manufacturing Year", "value": clean_display_value(data.get("year_of_manufacture") or data.get("manufacturing_year"))},
                {"label": "Seating Capacity", "value": clean_display_value(data.get("seating_capacity"))},
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
                {"label": "Vehicle IDV", "value": format_currency_inr(data.get("idv") or data.get("total_idv"))},
                {"label": "CNG/LPG IDV", "value": format_currency_inr(data.get("cng_idv"))},
                {"label": "Own Damage Premium", "value": format_currency_inr(data.get("basic_od_premium") or data.get("own_damage_premium") or data.get("od_premium"))},
                {"label": "Third Party Premium", "value": format_currency_inr(data.get("basic_tp_premium") or data.get("third_party_premium") or data.get("tp_premium"))},
                {"label": "Net Premium", "value": format_currency_inr(data.get("net_premium"))},
                {"label": "Add-on Premium", "value": format_currency_inr(data.get("addon_premium"))},
                {"label": "GST Amount", "value": format_currency_inr(data.get("tax") or data.get("gst") or data.get("gst_amount"))},
                {"label": "NCB", "value": clean_display_value(data.get("ncb") or data.get("ncb_percentage"))},
                {"label": "Total Premium", "value": format_currency_inr(data.get("total_premium")), "is_highlight": True},
            ],
        })

        # Important Dates Card
        sections.append({
            "title": "Important Dates",
            "icon": "📅",
            "fields": [
                {"label": "Booking Date", "value": clean_display_value(data.get("policy_booking_date"))},
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

    elif doc_type == DocumentType.PROPERTY_INSURANCE:
        sections.append({
            "title": "Insured Business Details",
            "icon": "🏢",
            "fields": [
                {"label": "Insured Business", "value": clean_display_value(data.get("insured_business") or data.get("customer_name"))},
                {"label": "Policy Number", "value": clean_display_value(data.get("policy_number"))},
                {"label": "Insurer", "value": clean_display_value(data.get("insurer"))},
                {"label": "Business / Occupancy", "value": clean_display_value(data.get("business_type") or data.get("occupancy"))},
                {"label": "Risk Address", "value": clean_display_value(data.get("address"))},
            ],
        })

        sections.append({
            "title": "Sum Insured & Coverage",
            "icon": "🛡️",
            "fields": [
                {"label": "Total Sum Insured", "value": format_currency_inr(data.get("total_sum_insured") or data.get("sum_insured")), "is_highlight": True},
                {"label": "Earthquake Cover", "value": clean_display_value(data.get("earthquake_cover"))},
                {"label": "Terrorism Cover", "value": clean_display_value(data.get("terrorism_cover"))},
            ],
        })

        sections.append({
            "title": "Financial Details",
            "icon": "💰",
            "fields": [
                {"label": "Base Premium", "value": format_currency_inr(data.get("base_premium") or data.get("premium"))},
                {"label": "Terrorism Premium", "value": format_currency_inr(data.get("terrorism_premium"))},
                {"label": "GST", "value": format_currency_inr(data.get("gst_amount") or data.get("gst") or data.get("tax"))},
                {"label": "Total Premium", "value": format_currency_inr(data.get("total_premium")), "is_highlight": True},
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

    elif doc_type == DocumentType.WORKMEN_COMPENSATION:
        sections.append({
            "title": "Employer & Policy Details",
            "icon": "👷",
            "fields": [
                {"label": "Employer Name", "value": clean_display_value(data.get("employer") or data.get("customer_name"))},
                {"label": "Policy Number", "value": clean_display_value(data.get("policy_number"))},
                {"label": "Insurer", "value": clean_display_value(data.get("insurer"))},
                {"label": "Business Activity", "value": clean_display_value(data.get("business_activity") or data.get("nature_of_work"))},
            ],
        })

        sections.append({
            "title": "Coverage & Workforce",
            "icon": "👥",
            "fields": [
                {"label": "Number of Employees", "value": clean_display_value(data.get("number_of_employees"))},
                {"label": "Sum Insured / Total Wages", "value": format_currency_inr(data.get("sum_insured")), "is_highlight": True},
                {"label": "Employment Location", "value": clean_display_value(data.get("employment_location") or data.get("address"))},
                {"label": "Occupational Disease Cover", "value": clean_display_value(data.get("occupational_disease_cover"))},
            ],
        })

        sections.append({
            "title": "Financial Details",
            "icon": "💰",
            "fields": [
                {"label": "Base Premium", "value": format_currency_inr(data.get("premium"))},
                {"label": "GST", "value": format_currency_inr(data.get("gst") or data.get("tax"))},
                {"label": "Total Premium", "value": format_currency_inr(data.get("total_premium")), "is_highlight": True},
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
