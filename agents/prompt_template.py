"""
Extraction Prompt Template Engine.
Provides complete connectivity between prompt specifications, LLM formatting,
and canonical structured field mapping.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from schemas.base import DocumentType
from utils.logger import get_logger

logger = get_logger("prompt_template")

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
EXTRACTION_PROMPT_FILE = PROMPTS_DIR / "extraction.txt"
EXTRACTION_SCHEMA_FILE = PROMPTS_DIR / "extraction_schema.json"

# Canonical 27 Motor Insurance extraction schema specifications
MOTOR_EXTRACTION_SCHEMA: Dict[str, str] = {
    "CUSTOMER_NAME": "Insured/policy holder name as printed (not agent name unless clearly labeled as customer).",
    "CUSTOMER_MOBILE": "Customer mobile as printed (10-digit India format if present).",
    "COMPANY_NAME": "Insurance company / insurer name as printed on the policy.",
    "AGENT_NAME": "Intermediary/agent/advisor name if explicitly labeled.",
    "AGENT_CODE": "Agent/intermediary code if explicitly labeled (often near agent details).",
    "CLASS_OF_VEHICLE": "Vehicle class/category , only 1 of this(private car, commercial, two wheeler, miscellaneous, bus, 3 wheeler) as labelled.",
    "INSURANCE_TYPE": "Policy type, only 1 of this(comprehensive/package, third party/liability, own damage) as labelled.",
    "POLICY_BOOKING_DATE": "Issue/booking/transaction date if explicitly labeled (not start date unless labeled as booking).",
    "POLICY_START_DATE": "Policy inception/start/period of insurance date as labeled.",
    "POLICY_END_DATE": "Policy expiry/end/period of insurance date as labeled.",
    "POLICY_NUMBER": "Policy number / certificate number as labeled (copy exactly).",
    "VEHICLE_REGISTRATION_NUMBER": "Registration number as printed (state code + series + number).",
    "TP_PREMIUM": "Total Third-party (TP)/Total Liability premium as labeled including addon, total of liability section, if has total premium then dont give basic premium, (numeric only).",
    "OD_PREMIUM": "Own damage (OD) premium as labeled (numeric only).",
    "NET_PREMIUM": "Net premium before GST/tax as labeled (numeric only).",
    "ADDON_PREMIUM": "Total add-on riders premium if shown (numeric only).",
    "GST_AMOUNT": "GST/IGST/CGST+SGST total as labeled (numeric only).",
    "TOTAL_PREMIUM": "Final payable/total premium as labeled (numeric only).",
    "TOTAL_IDV": "Total IDV / vehicle IDV as labeled (numeric only).",
    "CNG_IDV": "CNG/LPG kit IDV if present (numeric only).",
    "ENGINE_NUMBER": "Engine number as printed (copy exactly).",
    "CHASSIS_NUMBER": "Chassis/VIN as printed (copy exactly).",
    "YEAR_OF_MANUFACTURE": "Manufacturing year as labeled (4-digit year).",
    "MAKE": "Vehicle make/manufacturer as labeled.",
    "MODEL": "Vehicle model/variant as labeled.",
    "SEATING_CAPACITY": "Seating capacity as labeled (digits only).",
    "NCB": "NCB percentage/discount as labeled (include % if printed).",
}

# Bidirectional mapping from uppercase schema keys to internal canonical attributes
SCHEMA_TO_CANONICAL_MAP: Dict[str, List[str]] = {
    "CUSTOMER_NAME": ["customer_name", "insured_name"],
    "CUSTOMER_MOBILE": ["customer_mobile", "mobile"],
    "COMPANY_NAME": ["company_name", "insurance_company", "insurer"],
    "AGENT_NAME": ["agent_name"],
    "AGENT_CODE": ["agent_code"],
    "CLASS_OF_VEHICLE": ["class_of_vehicle", "vehicle_type"],
    "INSURANCE_TYPE": ["insurance_type", "policy_type"],
    "POLICY_BOOKING_DATE": ["policy_booking_date", "booking_date"],
    "POLICY_START_DATE": ["policy_start_date"],
    "POLICY_END_DATE": ["policy_end_date"],
    "POLICY_NUMBER": ["policy_number"],
    "VEHICLE_REGISTRATION_NUMBER": ["vehicle_registration_number", "registration_number"],
    "TP_PREMIUM": ["tp_premium", "third_party_premium"],
    "OD_PREMIUM": ["od_premium", "own_damage_premium"],
    "NET_PREMIUM": ["net_premium"],
    "ADDON_PREMIUM": ["addon_premium"],
    "GST_AMOUNT": ["gst_amount", "gst", "tax"],
    "TOTAL_PREMIUM": ["total_premium", "total_amount"],
    "TOTAL_IDV": ["total_idv", "idv"],
    "CNG_IDV": ["cng_idv"],
    "ENGINE_NUMBER": ["engine_number"],
    "CHASSIS_NUMBER": ["chassis_number"],
    "YEAR_OF_MANUFACTURE": ["year_of_manufacture", "manufacturing_year"],
    "MAKE": ["make", "vehicle_make"],
    "MODEL": ["model", "vehicle_model"],
    "SEATING_CAPACITY": ["seating_capacity"],
    "NCB": ["ncb", "ncb_percentage"],
}


class ExtractionPromptTemplate:
    """
    Manages prompt formatting, dynamic schema binding, and response parsing
    with complete connectivity into the extraction and validation pipeline.
    """

    def __init__(self, prompt_file: Optional[Path] = None, schema_file: Optional[Path] = None):
        self.prompt_file = prompt_file or EXTRACTION_PROMPT_FILE
        self.schema_file = schema_file or EXTRACTION_SCHEMA_FILE
        self._cached_schema: Optional[Dict[str, str]] = None
        self._cached_system_prompt: Optional[str] = None

    def get_schema(self) -> Dict[str, str]:
        """Loads and caches the target extraction schema."""
        if self._cached_schema is not None:
            return self._cached_schema

        if self.schema_file.is_file():
            try:
                data = json.loads(self.schema_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._cached_schema = data
                    return self._cached_schema
            except Exception as e:
                logger.warning(f"Failed to read schema file {self.schema_file}: {e}")

        self._cached_schema = dict(MOTOR_EXTRACTION_SCHEMA)
        return self._cached_schema

    def get_system_prompt(self) -> str:
        """Returns the system prompt with extraction rules and schema guidelines."""
        if self._cached_system_prompt is not None:
            return self._cached_system_prompt

        if self.prompt_file.is_file():
            try:
                content = self.prompt_file.read_text(encoding="utf-8").strip()
                if content:
                    self._cached_system_prompt = content
                    return self._cached_system_prompt
            except Exception as e:
                logger.warning(f"Failed to read prompt file {self.prompt_file}: {e}")

        # Fallback default prompt
        schema = self.get_schema()
        schema_json = json.dumps(schema, indent=2)
        default_prompt = (
            "You are a document extraction engine.\n\n"
            "Extract only information explicitly supported by the supplied document content.\n"
            "Never invent values.\n"
            "Never infer missing values as facts.\n"
            "Never merge unrelated fields.\n"
            "Respect field boundaries.\n\n"
            "Return structured data according to the supplied schema as a single valid JSON object.\n"
            "If a field is absent, return null.\n"
            "If a value is unreadable, return null rather than guessing.\n"
            "Do not include explanations or markdown formatting outside the JSON object.\n"
            "Do not include raw OCR noise.\n"
            "Do not add information from your general knowledge.\n\n"
            f"Extraction Schema:\n{schema_json}"
        )
        self._cached_system_prompt = default_prompt
        return default_prompt

    def format_user_prompt(
        self,
        text: str,
        doc_type: DocumentType,
        target_fields: Optional[List[str]] = None,
        max_chars: int = 6000,
    ) -> str:
        """
        Formats user message containing document text and specific field guidance.
        """
        schema = self.get_schema()
        if target_fields:
            # Match case-insensitively against schema keys
            filtered_schema = {}
            for tf in target_fields:
                tf_clean = tf.strip().upper()
                for sk, desc in schema.items():
                    if sk == tf_clean or sk.replace("_", "") == tf_clean.replace("_", "") or tf.lower() in [a.lower() for a in SCHEMA_TO_CANONICAL_MAP.get(sk, [])]:
                        filtered_schema[sk] = desc
                        break
                if tf_clean not in filtered_schema and tf in schema:
                    filtered_schema[tf] = schema[tf]
            active_schema = filtered_schema or schema
        else:
            active_schema = schema

        schema_snippet = json.dumps(active_schema, indent=2)
        doc_content = text[:max_chars].strip()

        return (
            f"Document Type: {doc_type.value}\n\n"
            f"Target Fields to Extract:\n{schema_snippet}\n\n"
            f"Document Text Content:\n\"\"\"\n{doc_content}\n\"\"\"\n\n"
            "Extract the target fields strictly according to the schema as a single valid JSON object. "
            "Use exact field names as keys. Return null for fields absent from the text."
        )

    def parse_llm_response(self, raw_content: str) -> Dict[str, Any]:
        """
        Robustly parses LLM JSON output, handles code fences, strips markdown,
        and synchronizes uppercase schema keys with canonical lowercase attributes.
        """
        if not raw_content or not str(raw_content).strip():
            return {}

        cleaned = raw_content.strip()

        # Remove markdown code block fences if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        # Find JSON object boundaries { ... }
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned = cleaned[start_idx : end_idx + 1]

        data: Dict[str, Any] = {}
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                data = parsed
        except Exception as e:
            logger.warning(f"Direct JSON parsing failed ({e}). Attempting regex fallback...")
            # Fallback regex key-value extraction
            for k in self.get_schema().keys():
                m = re.search(rf'"{k}"\s*:\s*(?:"([^"]*)"|([0-9\.]+)|(null|true|false))', cleaned, flags=re.IGNORECASE)
                if m:
                    val = m.group(1) or m.group(2)
                    if val is not None and val.lower() != "null":
                        data[k] = val

        # Synchronize keys: populate both original key and canonical lowercase aliases
        synchronized: Dict[str, Any] = {}
        for k, v in data.items():
            if v is None or str(v).strip().lower() in ("null", "none", ""):
                continue

            clean_val = str(v).strip() if isinstance(v, str) else v
            synchronized[k] = clean_val

            # Check mapping
            upper_k = str(k).strip().upper()
            if upper_k in SCHEMA_TO_CANONICAL_MAP:
                for canonical_alias in SCHEMA_TO_CANONICAL_MAP[upper_k]:
                    synchronized[canonical_alias] = clean_val
            else:
                # Lowercase fallback
                lower_k = str(k).strip().lower()
                synchronized[lower_k] = clean_val

        return synchronized


# Global singleton prompt template
extraction_template = ExtractionPromptTemplate()
