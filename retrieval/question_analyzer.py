"""
Question Analyzer: Classifies user queries and routes them to deterministic field resolvers or hybrid semantic RAG.
Rule: Do not send every simple field query through embeddings first.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import re


class QueryType(str, Enum):
    FIELD_QUERY = "FIELD_QUERY"
    MULTI_FIELD_QUERY = "MULTI_FIELD_QUERY"
    SEMANTIC_QUERY = "SEMANTIC_QUERY"


class AnalyzedQuestion:
    def __init__(
        self,
        raw_query: str,
        query_type: QueryType,
        target_field: Optional[str] = None,
        subfields: Optional[List[str]] = None,
        search_variants: Optional[List[str]] = None,
    ):
        self.raw_query = raw_query
        self.query_type = query_type
        self.target_field = target_field
        self.subfields = subfields or []
        self.search_variants = search_variants or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "query_type": self.query_type.value,
            "target_field": self.target_field,
            "subfields": self.subfields,
            "search_variants": self.search_variants,
        }


class QuestionAnalyzer:
    # Field trigger patterns
    FIELD_RULES = [
        ("policy_number", [
            r"\bpolicy\s*(?:no|number|id|#)\b",
            r"\bwhat\s+is\s+the\s+policy(?:\s*(?:no|number|id|#)|\s*$|\s*\?)",
            r"\bcertificate\s*(?:no|number)\b",
            r"\bcontract\s*number\b",
            r"\bconsumer\s*(?:no|number|id)\b",
            r"\bmeter\s*(?:no|number)\b",
        ]),
        ("customer_name", [
            r"\b(?:customer|insured|policyholder|proposer|consumer|client)\s*(?:'s)?\s*name\b",
            r"\bwho\s+is\s+(?:the\s+)?(?:customer|insured|policyholder|proposer)\b",
            r"\bname\s+of\s+(?:the\s+)?(?:insured|customer|proposer)\b",
        ]),
        ("policy_start_date", [
            r"\b(?:policy\s*)?(?:start|effective|commencement|inception|bill)\s*date\b",
            r"\bwhen\s+does\s+(?:the\s+)?policy\s*(?:start|commence|begin)\b",
            r"\bvalid\s*from\b",
            r"\beffective\s*from\b",
        ]),
        ("policy_end_date", [
            r"\b(?:policy\s*)?(?:end|expiry|expiration|due)\s*date\b",
            r"\bwhen\s+does\s+(?:the\s+)?(?:policy|bill)\s*(?:expire|end|due)\b",
            r"\bvalid\s*(?:to|until)\b",
            r"\blast\s*date\b",
        ]),
        ("policy_duration", [
            r"\b(?:policy\s*)?duration\b",
            r"\bhow\s+long\s+is\s+(?:the\s+)?policy\s*(?:valid|for)\b",
            r"\bpolicy\s*term\b",
            r"\bpolicy\s*period\b",
        ]),
        ("insurance_company", [
            r"\b(?:insurance\s*)?(?:company|insurer|provider|issuer)\b",
            r"\bwho\s+issued\s+(?:the\s+)?(?:policy|bill)\b",
            r"\bwhat\s+company\b",
            r"\bwhich\s+company\b",
        ]),
        ("net_premium", [
            r"\bnet\s*premium\b",
            r"\btotal\s*net\s*premium\b",
            r"\bpremium\s*before\s*tax\b",
            r"\bgross\s*premium\b",
        ]),
        ("basic_premium", [
            r"\bbasic\s*premium\b",
            r"\bbase\s*premium\b",
            r"\bown\s*damage\s*premium\b",
            r"\bthird\s*party\s*premium\b",
        ]),
        ("tax", [
            r"\b(?:what\s+is\s+the\s+)?gst\b",
            r"\bgoods\s*and\s*services\s*tax\b",
            r"\b(?:what\s+is\s+the\s+)?(?:tax|taxes)\b",
            r"\bigst\b",
            r"\bcgst\b",
            r"\bsgst\b",
            r"\belectricity\s*duty\b",
        ]),
        ("idv", [
            r"\bidv\b",
            r"\binsured\s*declared\s*value\b",
            r"\bvehicle\s*idv\b",
        ]),
        ("sum_insured", [
            r"\bsum\s*insured\b",
            r"\bsum\s*assured\b",
            r"\bcoverage\s*amount\b",
            r"\blimit\s*of\s*indemnity\b",
        ]),
        ("total_amount", [
            r"\btotal\s*(?:premium|amount|bill|payable)\b",
            r"\bwhat\s+is\s+the\s+(?:premium|total|amount|bill)\b",
            r"\bhow\s+much\s+(?:premium|is\s+the\s+bill|was\s+paid)\b",
            r"\bfinal\s*(?:premium|payable\s*amount|amount)\b",
            r"\bpremium\s*(?:amount|payable|paid)\b",
            r"\bamount\s*payable\b",
            r"\bcollected\s*amount\b",
            r"\bcollection\s*details\b",
        ]),
        ("registration_number", [
            r"\bregistration\s*(?:no|number)\b",
            r"\bregn?\s*no\b",
            r"\bvehicle\s*(?:number|reg)\b",
            r"\blicense\s*plate\b",
        ]),
        ("engine_number", [
            r"\bengine\s*(?:no|number)\b",
            r"\bengine\b",
        ]),
        ("chassis_number", [
            r"\bchassis\s*(?:no|number)\b",
            r"\bvin\b",
            r"\bchassis\b",
        ]),
        ("vehicle_details", [
            r"\bwhat\s+vehicle\b",
            r"\bwhich\s+vehicle\b",
            r"\bvehicle\s*(?:make|model|details)\b",
            r"\binsured\s+vehicle\b",
        ]),
        ("agent_name", [
            r"\b(?:agent|intermediary|advisor)\s*(?:'s)?\s*name\b",
            r"\bwho\s+is\s+(?:the\s+)?(?:agent|intermediary|advisor)\b",
        ]),
        ("agent_code", [
            r"\b(?:agent|intermediary|advisor)\s*(?:code|no|id)\b",
        ]),
        ("class_of_vehicle", [
            r"\b(?:class\s*of\s*vehicle|vehicle\s*class|vehicle\s*category)\b",
            r"\bwhat\s+class\s+is\s+the\s+vehicle\b",
        ]),
        ("insurance_type", [
            r"\b(?:insurance\s*type|policy\s*type|type\s*of\s*policy|type\s*of\s*insurance)\b",
        ]),
        ("policy_booking_date", [
            r"\b(?:booking|issue|transaction)\s*date\b",
            r"\bwhen\s+was\s+(?:the\s+)?policy\s*(?:booked|issued)\b",
        ]),
        ("customer_mobile", [
            r"\b(?:mobile|phone|contact)\s*(?:no|number)?\b",
            r"\bcustomer\s*mobile\b",
        ]),
        ("addon_premium", [
            r"\b(?:addon|add-on|rider)\s*premium\b",
        ]),
        ("cng_idv", [
            r"\b(?:cng|lpg)\s*(?:kit)?\s*idv\b",
        ]),
        ("seating_capacity", [
            r"\bseating\s*capacity\b",
            r"\bhow\s+many\s+seats\b",
            r"\bnumber\s+of\s+seats\b",
        ]),
        ("ncb", [
            r"\b(?:ncb|no\s*claim\s*bonus)(?:\s*(?:percentage|discount))?\b",
            r"\bwhat\s+is\s+the\s+ncb\b",
        ]),
        ("year_of_manufacture", [
            r"\b(?:manufacturing\s*year|year\s*of\s*manufacture|mfg\s*year)\b",
            r"\bwhen\s+was\s+(?:the\s+)?vehicle\s*manufactured\b",
        ]),
    ]

    SEMANTIC_TRIGGERS = [
        r"\bcover(?:age|ed)?\b",
        r"\bexclusion(?:s)?\b",
        r"\bcondition(?:s)?\b",
        r"\blimit(?:s|ation|ations)?\b",
        r"\bclause(?:s)?\b",
        r"\bdeductible(?:s)?\b",
        r"\bwhat\s+does\s+the\s+policy\s+cover\b",
        r"\bexplain\b",
        r"\bdescribe\b",
        r"\bsummarize\b",
    ]

    def analyze(self, query: str) -> AnalyzedQuestion:
        q_clean = query.strip().lower()

        # Check for multi-field inquiry
        if any(w in q_clean for w in ["all vehicle information", "vehicle summary", "vehicle info", "car details"]):
            return AnalyzedQuestion(
                raw_query=query,
                query_type=QueryType.MULTI_FIELD_QUERY,
                target_field="vehicle_details",
                subfields=["registration_number", "vehicle_make", "vehicle_model", "engine_number", "chassis_number"],
                search_variants=[query, "registration", "engine", "chassis", "vehicle make model"]
            )

        if any(w in q_clean for w in ["policy period and premium", "dates and amount", "policy summary"]):
            return AnalyzedQuestion(
                raw_query=query,
                query_type=QueryType.MULTI_FIELD_QUERY,
                target_field="period_and_amount",
                subfields=["policy_start_date", "policy_end_date", "policy_duration", "total_amount"],
                search_variants=[query, "period of insurance", "total premium", "from to"]
            )

        # Check for single field triggers
        for field_name, patterns in self.FIELD_RULES:
            for pat in patterns:
                if re.search(pat, q_clean):
                    variants = self._build_search_variants(field_name, query)
                    return AnalyzedQuestion(
                        raw_query=query,
                        query_type=QueryType.FIELD_QUERY,
                        target_field=field_name,
                        search_variants=variants
                    )

        # Check for semantic triggers
        for pat in self.SEMANTIC_TRIGGERS:
            if re.search(pat, q_clean):
                return AnalyzedQuestion(
                    raw_query=query,
                    query_type=QueryType.SEMANTIC_QUERY,
                    search_variants=[query]
                )

        # Default: Semantic Query with conservative variants
        return AnalyzedQuestion(
            raw_query=query,
            query_type=QueryType.SEMANTIC_QUERY,
            search_variants=[query]
        )

    def _build_search_variants(self, field_name: str, query: str) -> List[str]:
        variants = [query]
        if field_name == "policy_number":
            variants.extend(["policy number", "policy no", "certificate no", "policy #", "contract number"])
        elif field_name == "customer_name":
            variants.extend(["name of insured", "insured name", "customer name", "proposer", "policyholder"])
        elif field_name == "policy_start_date":
            variants.extend(["policy start date", "period from", "effective date", "commencement date"])
        elif field_name == "policy_end_date":
            variants.extend(["policy end date", "expiry date", "period to", "valid until"])
        elif field_name == "policy_duration":
            variants.extend(["period of insurance", "policy period", "from to", "duration"])
        elif field_name == "insurance_company":
            variants.extend(["insurance company", "insurer", "issued by", "underwritten by"])
        elif field_name == "net_premium":
            variants.extend(["net premium", "total net premium", "premium before tax", "gross premium"])
        elif field_name == "basic_premium":
            variants.extend(["basic premium", "base premium", "own damage premium", "third party premium"])
        elif field_name == "tax":
            variants.extend(["goods & services tax", "gst", "taxes", "igst", "cgst", "sgst"])
        elif field_name == "idv":
            variants.extend(["insured declared value", "idv", "vehicle idv"])
        elif field_name == "sum_insured":
            variants.extend(["total sum insured", "sum insured", "sum assured", "limit of indemnity"])
        elif field_name == "total_amount":
            variants.extend(["total premium", "total amount payable", "final premium", "total bill amount"])
        elif field_name == "registration_number":
            variants.extend(["registration number", "reg no", "vehicle registration"])
        elif field_name == "engine_number":
            variants.extend(["engine number", "engine no", "engine / motor no"])
        elif field_name == "chassis_number":
            variants.extend(["chassis number", "chassis no", "vin", "vin number"])
        elif field_name == "vehicle_details":
            variants.extend(["make and model", "vehicle make", "vehicle model"])
        elif field_name == "agent_name":
            variants.extend(["agent name", "intermediary name", "advisor name"])
        elif field_name == "agent_code":
            variants.extend(["agent code", "intermediary code", "advisor code"])
        elif field_name == "class_of_vehicle":
            variants.extend(["class of vehicle", "vehicle class", "vehicle category"])
        elif field_name == "insurance_type":
            variants.extend(["policy type", "insurance type", "type of policy"])
        elif field_name == "policy_booking_date":
            variants.extend(["booking date", "issue date", "transaction date"])
        elif field_name == "customer_mobile":
            variants.extend(["mobile number", "phone number", "contact number"])
        elif field_name == "addon_premium":
            variants.extend(["addon premium", "add-on premium", "rider premium"])
        elif field_name == "cng_idv":
            variants.extend(["cng idv", "lpg idv", "cng kit idv"])
        elif field_name == "seating_capacity":
            variants.extend(["seating capacity", "seats", "number of seats"])
        elif field_name == "ncb":
            variants.extend(["no claim bonus", "ncb", "ncb discount", "ncb percentage"])
        elif field_name == "year_of_manufacture":
            variants.extend(["manufacturing year", "year of manufacture", "mfg year"])
        return variants


# Global Question Analyzer Singleton
question_analyzer = QuestionAnalyzer()
