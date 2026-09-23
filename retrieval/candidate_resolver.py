"""
Candidate Resolver: Central, authoritative candidate ranking and resolution engine.
Arbitrates between multiple extraction mechanisms (tables, spatial layout, dedicated
resolvers, deterministic regex, and LLMs) to select exactly one grounded value per canonical field.
Enforces hard validation, eliminates duplicate aliases, and prevents non-serializable objects from escaping.
"""

import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Set
from schemas.base import DocumentType, SourceEvidence
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
from utils.logger import get_logger

logger = get_logger("candidate_resolver")

# -------------------------------------------------------------
# Canonical Schema Field Definitions
# -------------------------------------------------------------
CANONICAL_MOTOR_FIELDS = [
    "customer_name",
    "customer_mobile",
    "insurance_company",
    "agent_name",
    "agent_code",
    "policy_number",
    "secondary_policy_number",
    "previous_policy_number",
    "previous_insurer",
    "policy_booking_date",
    "policy_start_date",
    "policy_end_date",
    "policy_duration",
    "registration_number",
    "vehicle_make",
    "vehicle_model",
    "engine_number",
    "chassis_number",
    "year_of_manufacture",
    "seating_capacity",
    "class_of_vehicle",
    "insurance_type",
    "idv",
    "cng_idv",
    "ncb",
    "basic_od_premium",
    "basic_tp_premium",
    "net_premium",
    "addon_premium",
    "tax",
    "total_premium",
    "insured_address",
    "email",
    "rto",
    "coverage",
    "nominee",
]

CANONICAL_ELECTRICITY_FIELDS = [
    "customer_name",
    "consumer_number",
    "meter_number",
    "bill_number",
    "bill_date",
    "due_date",
    "previous_reading",
    "current_reading",
    "consumption",
    "fixed_charge",
    "energy_charge",
    "fuel_charge",
    "electricity_duty",
    "total_bill",
    "net_bill_amount",
    "tariff",
    "phase",
    "electricity_provider",
    "billing_units",
    "address",
]

CANONICAL_HEALTH_FIELDS = [
    "customer_name",
    "policyholder_name",
    "customer_mobile",
    "insurance_company",
    "policy_number",
    "proposal_number",
    "customer_id",
    "address",
    "email",
    "product_name",
    "plan_name",
    "policy_type",
    "policy_start_date",
    "policy_end_date",
    "sum_insured",
    "cumulative_bonus",
    "deductible",
    "nominee",
    "family_members",
    "net_premium",
    "tax",
    "total_premium",
    "hospitalization_cover",
]

CANONICAL_PROPERTY_FIELDS = [
    "customer_name",
    "policy_number",
    "insured_business",
    "business_type",
    "insurance_company",
    "address",
    "gst",
    "policy_start_date",
    "policy_end_date",
    "risk_locations",
    "sum_insured_breakdown",
    "total_sum_insured",
    "sum_insured",
    "base_premium",
    "terrorism_premium",
    "tax",
    "total_premium",
    "earthquake_cover",
    "terrorism_cover",
    "deductibles",
]

CANONICAL_WORKMEN_FIELDS = [
    "customer_name",
    "employer",
    "policy_number",
    "insurance_company",
    "business_activity",
    "nature_of_work",
    "address",
    "gst",
    "policy_start_date",
    "policy_end_date",
    "territory",
    "jurisdiction",
    "sum_insured",
    "number_of_employees",
    "employee_details",
    "employment_location",
    "medical_expenses_limit",
    "occupational_disease_cover",
    "contractor_cover",
    "subcontractor_cover",
    "tax",
    "total_premium",
]

CANONICAL_GENERIC_FIELDS = [
    "customer_name",
    "policy_number",
    "company_name",
    "date",
    "total_amount",
]

SCHEMA_BY_DOC_TYPE: Dict[DocumentType, List[str]] = {
    DocumentType.MOTOR_INSURANCE: CANONICAL_MOTOR_FIELDS,
    DocumentType.ELECTRICITY_BILL: CANONICAL_ELECTRICITY_FIELDS,
    DocumentType.HEALTH_INSURANCE: CANONICAL_HEALTH_FIELDS,
    DocumentType.PROPERTY_INSURANCE: CANONICAL_PROPERTY_FIELDS,
    DocumentType.WORKMEN_COMPENSATION: CANONICAL_WORKMEN_FIELDS,
    DocumentType.INVOICE: CANONICAL_GENERIC_FIELDS,
    DocumentType.FINANCIAL_DOCUMENT: CANONICAL_GENERIC_FIELDS,
    DocumentType.GENERIC_DOCUMENT: CANONICAL_GENERIC_FIELDS,
}

# -------------------------------------------------------------
# Raw Alias to Canonical Mapping
# -------------------------------------------------------------
ALIAS_TO_CANONICAL: Dict[str, str] = {
    # Customer name
    "customer_name": "customer_name",
    "insured_name": "customer_name",
    "customer": "customer_name",
    "proposer_name": "customer_name",
    "customer_full_name": "customer_name",
    "customer_name": "customer_name",
    "primary_party_name": "customer_name",
    "policyholder_name": "customer_name",

    # Mobile
    "customer_mobile": "customer_mobile",
    "mobile": "customer_mobile",
    "phone": "customer_mobile",
    "contact_no": "customer_mobile",

    # Insurer / Provider
    "insurance_company": "insurance_company",
    "insurer": "insurance_company",
    "company_name": "insurance_company",
    "electricity_provider": "electricity_provider",

    # Agent
    "agent_name": "agent_name",
    "intermediary_name": "agent_name",
    "advisor_name": "agent_name",
    "agent_code": "agent_code",
    "intermediary_code": "agent_code",
    "advisor_code": "agent_code",

    # Policy / Consumer / Reference numbers
    "policy_number": "policy_number",
    "policy_no": "policy_number",
    "policy_certificate_no": "policy_number",
    "certificate_no": "policy_number",
    "contract_number": "policy_number",
    "secondary_policy_number": "secondary_policy_number",
    "consumer_number": "consumer_number",
    "ca_no": "consumer_number",
    "meter_number": "meter_number",
    "meter_no": "meter_number",
    "bill_number": "bill_number",
    "bill_no": "bill_number",

    # Dates
    "policy_booking_date": "policy_booking_date",
    "booking_date": "policy_booking_date",
    "issue_date": "policy_booking_date",
    "issuance_date": "policy_booking_date",
    "policy_start_date": "policy_start_date",
    "start_date": "policy_start_date",
    "period_from": "policy_start_date",
    "policy_end_date": "policy_end_date",
    "end_date": "policy_end_date",
    "period_to": "policy_end_date",
    "policy_duration": "policy_duration",
    "policy_term": "policy_duration",
    "bill_date": "bill_date",
    "due_date": "due_date",
    "date": "date",

    # Vehicle
    "registration_number": "registration_number",
    "vehicle_registration_number": "registration_number",
    "reg_no": "registration_number",
    "regn_no": "registration_number",
    "vehicle_make": "vehicle_make",
    "make": "vehicle_make",
    "manufacturer": "vehicle_make",
    "vehicle_model": "vehicle_model",
    "model": "vehicle_model",
    "variant": "vehicle_model",
    "engine_number": "engine_number",
    "engine_no": "engine_number",
    "chassis_number": "chassis_number",
    "chassis_no": "chassis_number",
    "vin": "chassis_number",
    "year_of_manufacture": "year_of_manufacture",
    "manufacturing_year": "year_of_manufacture",
    "seating_capacity": "seating_capacity",
    "seats": "seating_capacity",
    "class_of_vehicle": "class_of_vehicle",
    "vehicle_class": "class_of_vehicle",
    "vehicle_type": "class_of_vehicle",
    "insurance_type": "insurance_type",
    "policy_type": "insurance_type",

    # Financials
    "idv": "idv",
    "total_idv": "idv",
    "vehicle_idv": "idv",
    "cng_idv": "cng_idv",
    "cng_value": "cng_idv",
    "ncb": "ncb",
    "ncb_percentage": "ncb",
    "no_claim_bonus": "ncb",
    "no_claim_discount": "ncb",
    "basic_od_premium": "basic_od_premium",
    "own_damage_premium": "basic_od_premium",
    "od_premium": "basic_od_premium",
    "basic_od": "basic_od_premium",
    "basic_tp_premium": "basic_tp_premium",
    "third_party_premium": "basic_tp_premium",
    "tp_premium": "basic_tp_premium",
    "basic_tp": "basic_tp_premium",
    "net_premium": "net_premium",
    "addon_premium": "addon_premium",
    "tax": "tax",
    "gst": "tax",
    "gst_amount": "tax",
    "total_gst": "tax",
    "igst": "tax",
    "cgst": "tax",
    "sgst": "tax",
    "integrated_tax": "tax",
    "total_premium": "total_premium",
    "total_amount": "total_premium",
    "gross_premium": "total_premium",
    "final_premium": "total_premium",
    "amount_payable": "total_premium",
    "premium_payable": "total_premium",
    "total_premium_payable": "total_premium",
    "premium": "total_premium",
    "sum_insured": "sum_insured",
    "total_sum_insured": "total_sum_insured",
    "total_bill": "total_bill",
    "net_bill_amount": "net_bill_amount",
    "billing_units": "billing_units",

    # Domain specific fields - Property, Health, Workmen, Motor, Electricity
    "insured_business": "insured_business",
    "business_type": "business_type",
    "business_activity": "business_activity",
    "nature_of_work": "nature_of_work",
    "employer": "employer",
    "territory": "territory",
    "jurisdiction": "jurisdiction",
    "number_of_employees": "number_of_employees",
    "employee_details": "employee_details",
    "employment_location": "employment_location",
    "medical_expenses_limit": "medical_expenses_limit",
    "occupational_disease_cover": "occupational_disease_cover",
    "contractor_cover": "contractor_cover",
    "subcontractor_cover": "subcontractor_cover",
    "product_name": "product_name",
    "plan_name": "plan_name",
    "family_members": "family_members",
    "hospitalization_cover": "hospitalization_cover",
    "deductible": "deductible",
    "cumulative_bonus": "cumulative_bonus",
    "nominee": "nominee",
    "proposal_number": "proposal_number",
    "customer_id": "customer_id",
    "risk_locations": "risk_locations",
    "sum_insured_breakdown": "sum_insured_breakdown",
    "base_premium": "base_premium",
    "terrorism_premium": "terrorism_premium",
    "earthquake_cover": "earthquake_cover",
    "terrorism_cover": "terrorism_cover",
    "previous_reading": "previous_reading",
    "current_reading": "current_reading",
    "consumption": "consumption",
    "fixed_charge": "fixed_charge",
    "energy_charge": "energy_charge",
    "fuel_charge": "fuel_charge",
    "electricity_duty": "electricity_duty",
    "tariff": "tariff",
    "phase": "phase",
    "previous_policy_number": "previous_policy_number",
    "previous_insurer": "previous_insurer",
    "coverage": "coverage",
    "rto": "rto",
    "insured_address": "insured_address",
    "address": "address",
    "email": "email",
    "branch": "branch",
    "branch_office": "branch",
    "branch_address": "branch",
}


@dataclass
class FieldCandidate:
    field_name: str
    value: Any
    exact_label: str = ""
    raw_evidence: str = ""
    page: int = 1
    bbox: Optional[Tuple[float, float, float, float]] = None
    method: str = "unknown"
    confidence: float = 0.50
    priority_tier: int = 1  # 4=Table/Layout/Stacked, 3=Dedicated Resolver, 2=Deterministic Regex, 1=LLM
    source_context: str = ""


class CandidateResolver:
    """
    Collects, validates, scores, and resolves field candidates into a single
    authoritative set of canonical fields.
    """
    def __init__(self, doc_type: DocumentType):
        self.doc_type = doc_type
        self.target_fields = SCHEMA_BY_DOC_TYPE.get(doc_type, CANONICAL_GENERIC_FIELDS)
        self.candidates_by_field: Dict[str, List[FieldCandidate]] = {}

    def add_candidate(
        self,
        raw_field: str,
        value: Any,
        exact_label: str = "",
        raw_evidence: str = "",
        page: int = 1,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        method: str = "unknown",
        confidence: float = 0.50,
        priority_tier: int = 1,
        source_context: str = ""
    ):
        """Adds a new candidate for evaluation, mapping internal aliases to canonical names."""
        if value is None:
            return
        v_str = str(value).strip()
        if not v_str or v_str.lower() in ("none", "null", "n/a", "na"):
            return

        # Canonicalize field name (case-insensitive lookup)
        clean_raw = raw_field.strip().lower()
        canonical_field = ALIAS_TO_CANONICAL.get(clean_raw, clean_raw)

        # Domain-specific field mapping adjustments
        if self.doc_type == DocumentType.ELECTRICITY_BILL:
            if canonical_field == "policy_number":
                canonical_field = "consumer_number"
            elif canonical_field == "policy_start_date":
                canonical_field = "bill_date"
            elif canonical_field == "policy_end_date":
                canonical_field = "due_date"
            elif canonical_field == "total_premium":
                canonical_field = "total_bill"
            elif canonical_field == "insurance_company":
                canonical_field = "electricity_provider"

        cand = FieldCandidate(
            field_name=canonical_field,
            value=v_str,
            exact_label=exact_label or raw_field,
            raw_evidence=raw_evidence or v_str,
            page=page,
            bbox=bbox,
            method=method,
            confidence=confidence,
            priority_tier=priority_tier,
            source_context=source_context
        )

        if canonical_field not in self.candidates_by_field:
            self.candidates_by_field[canonical_field] = []
        self.candidates_by_field[canonical_field].append(cand)

    def load_from_raw_extraction(
        self,
        raw_data: Dict[str, Any],
        evidence_candidates: Optional[Dict[str, Any]] = None
    ):
        """
        Convenience method to ingest all candidates from raw_data and evidence_candidates
        with appropriate priority tiers.
        """
        ev_cands = evidence_candidates or raw_data.get("_evidence_candidates", {})

        # 1. Ingest dedicated resolver / table / spatial candidates (Tier 3-4)
        for field, cand in ev_cands.items():
            if hasattr(cand, "value") and cand.value is not None:
                method = getattr(cand, "method", "unknown")
                is_tier_4 = any(k in method for k in ["table_", "spatial_", "stacked_"])
                priority = 4 if is_tier_4 else 3
                self.add_candidate(
                    raw_field=field,
                    value=cand.value,
                    exact_label=getattr(cand, "exact_label", "") or field,
                    raw_evidence=getattr(cand, "raw_evidence", "") or str(cand.value),
                    page=getattr(cand, "page", 1),
                    bbox=getattr(cand, "bbox", None),
                    method=method,
                    confidence=getattr(cand, "confidence", 0.95),
                    priority_tier=priority
                )

        # 2. Ingest raw extraction keys (Tier 2 if regex, Tier 1 if LLM)
        for k, v in raw_data.items():
            if k.startswith("_") or v is None:
                continue
            if k in ev_cands:
                continue
            is_llm = "llm" in k.lower()
            priority = 1 if is_llm else 2
            conf = 0.60 if is_llm else 0.80
            method = "llm" if is_llm else "deterministic_regex"
            self.add_candidate(
                raw_field=k,
                value=v,
                exact_label=k,
                raw_evidence=str(v),
                page=1,
                method=method,
                confidence=conf,
                priority_tier=priority
            )

    def resolve_all(
        self,
        page_texts: Optional[Dict[int, str]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, SourceEvidence]]:
        """
        Executes candidate resolution across all canonical fields.
        Returns:
            (canonical_structured_data, evidence_map)
        """
        canonical_data: Dict[str, Any] = {}
        evidence_map: Dict[str, SourceEvidence] = {}

        # 1. Process all fields that have candidates
        all_fields = list(self.candidates_by_field.keys())
        for field_name in all_fields:
            cands = self.candidates_by_field[field_name]
            best_cand = self._select_winning_candidate(field_name, cands)
            if best_cand:
                norm_val = self._normalize_candidate_value(field_name, best_cand.value)
                if norm_val is not None:
                    canonical_data[field_name] = norm_val
                    evidence_map[field_name] = SourceEvidence(
                        field=field_name,
                        value=norm_val,
                        exact_label=best_cand.exact_label,
                        page=best_cand.page,
                        bbox=best_cand.bbox,
                        source=f"Page {best_cand.page}",
                        evidence=best_cand.raw_evidence,
                        raw_evidence=best_cand.raw_evidence,
                        method=best_cand.method,
                        confidence=best_cand.confidence,
                    )

        # 2. Date consistency check (Requirement 4)
        s_date = canonical_data.get("policy_start_date")
        e_date = canonical_data.get("policy_end_date")
        if s_date and e_date and s_date == e_date:
            logger.warning(f"Conflicting identical start and end dates detected ({s_date}). Investigating candidates.")
            start_cands = self.candidates_by_field.get("policy_start_date", [])
            for sc in start_cands:
                val = self._normalize_candidate_value("policy_start_date", sc.value)
                if val and val != e_date:
                    canonical_data["policy_start_date"] = val
                    if "policy_start_date" in evidence_map:
                        evidence_map["policy_start_date"].value = val
                    break

        # 3. Model vs Make consistency check (Requirement 3, 10)
        # If vehicle_model starts with vehicle_make, strip make from model
        v_make = canonical_data.get("vehicle_make")
        v_model = canonical_data.get("vehicle_model")
        if v_make and v_model and isinstance(v_make, str) and isinstance(v_model, str):
            if v_model.upper().startswith(v_make.upper() + " "):
                stripped_model = v_model[len(v_make):].strip()
                if stripped_model:
                    canonical_data["vehicle_model"] = stripped_model
                    if "vehicle_model" in evidence_map:
                        evidence_map["vehicle_model"].value = stripped_model

        # Property and Health insurance sum_insured cross-sync
        if "total_sum_insured" in self.target_fields or "sum_insured" in self.target_fields:
            if not canonical_data.get("total_sum_insured") and canonical_data.get("sum_insured"):
                canonical_data["total_sum_insured"] = canonical_data["sum_insured"]
            elif not canonical_data.get("sum_insured") and canonical_data.get("total_sum_insured"):
                canonical_data["sum_insured"] = canonical_data["total_sum_insured"]

        # Health insurance customer_name / policyholder_name synchronization
        if "policyholder_name" in self.target_fields or "customer_name" in self.target_fields:
            if not canonical_data.get("policyholder_name") and canonical_data.get("customer_name"):
                canonical_data["policyholder_name"] = canonical_data["customer_name"]
            elif not canonical_data.get("customer_name") and canonical_data.get("policyholder_name"):
                canonical_data["customer_name"] = canonical_data["policyholder_name"]

        # 4. Filter output to authoritative schema for current DocumentType
        final_structured: Dict[str, Any] = {}
        for f in self.target_fields:
            if f in canonical_data:
                final_structured[f] = canonical_data[f]
            else:
                final_structured[f] = None

        # Keep any secondary policy numbers if discovered
        if canonical_data.get("secondary_policy_number"):
            final_structured["secondary_policy_number"] = canonical_data["secondary_policy_number"]

        # Preserve any valid domain fields extracted in canonical_data
        for f, val in canonical_data.items():
            if f not in final_structured and val is not None and not f.startswith("_"):
                final_structured[f] = val

        return final_structured, evidence_map

    def _select_winning_candidate(
        self,
        field_name: str,
        candidates: List[FieldCandidate]
    ) -> Optional[FieldCandidate]:
        """Ranks candidates using priority tier, method quality, and strict validation."""
        valid_cands: List[Tuple[float, FieldCandidate]] = []

        for cand in candidates:
            # Step A: Enforce Hard Negative Validation
            if not self._passes_negative_validation(field_name, cand.value):
                logger.debug(f"Rejected candidate for {field_name}: {cand.value} (method: {cand.method})")
                continue

            # Step B: Compute composite rank score
            score = self._compute_candidate_score(field_name, cand)
            valid_cands.append((score, cand))

        if not valid_cands:
            return None

        # Sort descending by composite score
        valid_cands.sort(key=lambda x: x[0], reverse=True)
        return valid_cands[0][1]

    def _compute_candidate_score(self, field_name: str, cand: FieldCandidate) -> float:
        """Calculates a numerical ranking score for candidate selection."""
        # Base priority tier (4=Table/Layout/Stacked, 3=Dedicated Resolver, 2=Deterministic Regex, 1=LLM)
        score = cand.priority_tier * 100.0

        # Method specific quality bonus
        if "table_cell" in cand.method:
            score += 30.0
        elif "stacked_above_label" in cand.method:
            score += 28.0
        elif "header_identity" in cand.method or "table_header_block" in cand.method:
            score += 25.0
        elif "spatial_right" in cand.method:
            score += 25.0
        elif "spatial_below" in cand.method:
            score += 15.0
        elif "inline_delimiter" in cand.method:
            score += 18.0
        elif "dedicated_resolver" in cand.method:
            score += 20.0
        elif "regex" in cand.method:
            score += 10.0
        elif "llm" in cand.method:
            score += 5.0

        # Exact label bonus: explicit labels score higher than generic keyword matches
        lbl_lower = cand.exact_label.lower()
        if field_name == "customer_name" and any(k in lbl_lower for k in ("insured", "customer", "proposer", "name")):
            score += 15.0
        elif field_name == "vehicle_model" and ("model" in lbl_lower or "variant" in lbl_lower):
            score += 15.0
        elif field_name == "vehicle_make" and "make" in lbl_lower:
            score += 15.0
        elif field_name == "policy_number" and ("policy" in lbl_lower or "certificate" in lbl_lower):
            score += 15.0

        # Page role bonus: Identity / header fields prefer Page 1
        if field_name in ("customer_name", "policy_number", "insurance_company", "registration_number", "vehicle_make", "vehicle_model"):
            if cand.page == 1:
                score += 20.0
            elif cand.page == 2:
                score += 5.0
            else:
                score -= 20.0

        # Direct confidence contribution
        score += cand.confidence * 20.0

        return score

    def _passes_negative_validation(self, field_name: str, value: Any) -> bool:
        """Enforces domain-specific negative constraints to guarantee zero hallucination."""
        val_str = str(value).strip()
        val_upper = val_str.upper()

        if field_name == "vehicle_model":
            # Requirement 3, 10
            disallowed = [
                "OF VEHICLE", "VEHICLE", "MAKE", "MODEL", "TYPE OF BODY", "GVW",
                "SUB-TYPE", "VARIANT", "CHASSIS", "CHASSIS NO", "CHASSIS NO.",
                "ENGINE", "ENGINE NO", "ENGINE NO.", "REGISTRATION", "REGISTRATION NO",
                "REG NO", "SEATING", "CAPACITY", "NONE", "NA"
            ]
            if val_upper in disallowed or val_upper.startswith("OF VEHICLE"):
                return False
            # Prevent column contamination like "JUPITER-ZX Period Insurance of From..."
            if any(term in val_upper for term in [
                "PERIOD INSURANCE", "PERIOD OF INSURANCE", "FROM 10", "TO 09", "EXPIRY",
                "CHASSIS", "ENGINE", "REGISTRATION", "SEATING", "CAPACITY", "FASTAG", "ODOMETER", "INVOICE", "PARTNER"
            ]):
                return False
            if len(val_str) < 2 or len(val_str) > 40:
                return False

        elif field_name == "vehicle_make":
            disallowed = ["MAKE", "MODEL", "VEHICLE", "DETAILS", "OF VEHICLE", "CHASSIS", "ENGINE"]
            if val_upper in disallowed:
                return False
            if len(val_str) < 2 or len(val_str) > 30:
                return False

        elif field_name == "customer_name":
            # Requirement 7: Structural customer name validation
            disallowed_exact = {
                "PERSON", "CUSTOMER", "CUSTOMER ID", "CUSTOMER NO", "CUSTOMER NUMBER",
                "POLICYHOLDER", "POLICYHOLDER NAME", "POLICY HOLDER", "POLICY HOLDER NAME",
                "NAME", "NAME OF INSURED PERSON", "NAME OF INSURED", "NAME OF THE INSURED",
                "NAME OF PROPOSER", "INSURED PERSON", "INSURED PERSON(S)", "INSURED PERSONS",
                "SELF", "SPOUSE", "DAUGHTER", "SON", "FATHER", "MOTHER", "RELATIONSHIP", "RELATION",
                "DATE OF BIRTH", "DOB", "GENDER", "MALE", "FEMALE", "MEMBER", "MEMBER ID",
                "POLICY NUMBER", "POLICY NO", "PLAN NAME", "PRODUCT NAME", "ADDRESS", "PERMANENT ADDRESS",
                "CHASSIS", "CHASSIS NO", "CHASSIS NUMBER", "ENGINE NO", "ENGINE NUMBER",
                "REGISTRATION NO", "REGISTRATION NUMBER", "REG NO", "VEHICLE NO", "MAKE", "MODEL",
                "PERIOD OF INSURANCE", "AGE"
            }
            if val_upper in disallowed_exact:
                return False
            for dis in disallowed_exact:
                if val_upper == dis or val_upper.startswith(dis + ":") or val_upper.startswith(dis + " "):
                    if len(val_upper.replace(dis, "").strip(" :.-#")) < 3:
                        return False
            boilerplate = [
                "DEFENCE", "SETTLEMENT", "CONDITIONS", "LIMITATIONS", "POLICY",
                "PREMIUM", "VEHICLE", "COURT", "LIABILITY", "PROCEEDINGS",
                "SCHEDULE", "IMPORTANT", "NOTICE", "CERTIFICATE", "INSURANCE",
                "BRANCH", "REGISTERED", "OFFICE", "BROKER", "AGENT", "BANK",
                "POLICY BAZAAR", "POLICYBAZAAR", "COVERFOX",
                "CUSTOMER ID", "CUSTOMER NO", "POLICY NUMBER", "POLICY NO",
                "PLAN NAME", "ADDRESS", "CHASSIS", "ENGINE NO", "REGISTRATION NO", "REG NO"
            ]
            if any(bad in val_upper for bad in boilerplate):
                return False
            # Must not contain digits
            if any(char.isdigit() for char in val_str):
                return False
            # Must contain at least two alphabetic characters
            if len(re.sub(r"[^a-zA-Z]", "", val_str)) < 2:
                return False
            # Plausible length for person / company name
            if len(val_str) < 3 or len(val_str) > 60:
                return False

        elif field_name in ("policy_number", "secondary_policy_number"):
            # Requirement 4, 8
            disallowed = [
                "PAYABLE", "PREVIOUS", "BENEFIT", "UNLESS", "POLICY",
                "SCHEDULE", "CERTIFICATE", "CONDITIONS", "REGISTRATION DETAILS",
                "RC COPY", "PREMIUM CHEQUE", "CHEQUE", "SUPPORT PLUS", "GOLD PLAN",
                "SILVER PLAN", "PLATINUM PLAN", "PACKAGE POLICY", "CUSTOMER ID"
            ]
            if any(bad in val_upper for bad in disallowed):
                return False
            # Policy number must contain at least one digit
            if not any(char.isdigit() for char in val_str):
                return False
            # Reject checklist bullets e.g. "b. Registration Details..."
            if re.match(r"^[a-zA-Z0-9][\.\)]\s+", val_str):
                return False
            if len(val_str) < 4 or len(val_str) > 45:
                return False

        elif field_name in ("engine_number", "chassis_number"):
            disallowed = [
                "CHASSIS", "CHASSIS NO", "CHASSIS NO.", "ENGINE", "ENGINE NO",
                "ENGINE NO.", "REGISTRATION", "REGISTRATION NO", "REG NO", "MODEL", "MAKE"
            ]
            if val_upper in disallowed:
                return False
            if any(val_upper.startswith(lbl) and len(val_upper) == len(lbl) for lbl in disallowed):
                return False
            if len(val_str) < 4 or len(val_str) > 35:
                return False

        elif field_name in ("insurance_company", "electricity_provider"):
            # Requirement 9: Do not confuse broker, agent, or bank with insurer
            if any(bad in val_upper for bad in ["BROKER", "INTERMEDIARY", "WEB AGGREGATOR", "POLICYBAZAAR", "POLICY BAZAAR", "COVERFOX", "BANK", "FINANCIER"]):
                return False
            if len(val_str) < 4 or len(val_str) > 80:
                return False

        elif field_name in ("policy_start_date", "policy_end_date", "policy_booking_date", "bill_date", "due_date"):
            # Must be a valid parseable date
            d = parse_and_normalize_date(val_str)
            if not d:
                return False

        elif field_name in ("idv", "basic_od_premium", "basic_tp_premium", "net_premium", "total_premium", "sum_insured", "total_amount"):
            clean_no_curr = re.sub(r"(?i)rs\.?|inr|₹|/-", "", val_str).strip()
            if any(c.isalpha() for c in clean_no_curr):
                return False
            num = parse_currency_amount(val_str)
            if num is None or num <= 0:
                return False
            if re.search(r"\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b", val_str):
                return False

        elif field_name in ("cng_idv", "tax"):
            clean_no_curr = re.sub(r"(?i)rs\.?|inr|₹|/-", "", val_str).strip()
            if any(c.isalpha() for c in clean_no_curr):
                return False
            num = parse_currency_amount(val_str)
            if num is None or num < 0:
                return False

        elif field_name == "ncb":
            clean_no_pct = re.sub(r"(?i)rs\.?|inr|₹|/-|%", "", val_str).strip()
            if any(c.isalpha() for c in clean_no_pct):
                return False
            if not any(c.isdigit() for c in val_str):
                return False

        return True

    def _normalize_candidate_value(self, field_name: str, value: Any) -> Any:
        """Produces clean canonical JSON-serializable primitives."""
        v_str = str(value).strip()
        if not v_str:
            return None

        if field_name == "customer_name":
            return clean_customer_name(v_str)
        elif field_name == "registration_number":
            return clean_registration_number(v_str)
        elif field_name in ("policy_number", "secondary_policy_number", "consumer_number", "meter_number", "bill_number"):
            return clean_policy_number(v_str)
        elif field_name == "customer_mobile":
            return clean_phone_number(v_str)
        elif field_name in ("policy_start_date", "policy_end_date", "policy_booking_date", "bill_date", "due_date", "date"):
            return parse_and_normalize_date(v_str)
        elif field_name in ("idv", "cng_idv", "basic_od_premium", "basic_tp_premium", "net_premium", "tax", "total_premium", "total_bill", "net_bill_amount", "sum_insured", "total_sum_insured", "base_premium", "terrorism_premium", "total_amount"):
            return parse_currency_amount(v_str)
        elif field_name == "number_of_employees":
            try:
                digits = re.sub(r"[^\d]", "", v_str)
                return int(digits) if digits else None
            except Exception:
                return None
        elif field_name == "ncb":
            return clean_ncb(v_str)
        elif field_name == "seating_capacity":
            return clean_seating_capacity(v_str)
        elif field_name in ("year_of_manufacture", "manufacturing_year"):
            return clean_year(v_str)
        elif field_name in ("engine_number", "engine_no"):
            return clean_engine_number(v_str)
        elif field_name in ("chassis_number", "chassis_no", "vin"):
            return clean_chassis_number(v_str)
        elif field_name == "class_of_vehicle":
            return clean_vehicle_class(v_str)
        elif field_name == "insurance_type":
            return clean_insurance_type(v_str)
        else:
            cleaned = " ".join(v_str.split())
            return cleaned if cleaned else None
