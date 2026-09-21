"""
Document Classifier: Content-based classification engine.
Strictly ignores filenames to avoid deception (e.g. Electricity bill named PANCARD.pdf).
Combines weighted lexical signal analysis with semantic fallback.
"""

import re
from typing import Tuple, Optional
from schemas.base import DocumentType, MotorSubtype
from utils.logger import get_logger

logger = get_logger("classifier")

# Lexical fingerprint patterns for deterministic classification
CLASSIFICATION_RULES = {
    DocumentType.ELECTRICITY_BILL: {
        "keywords": [
            "electricity", "power distribution", "discom", "consumer no", "meter no",
            "kwh", "units consumed", "tariff", "bill amount", "connected load",
            "reading date", "due date", "fuel surcharge", "power factor", "guvnl",
            "bescom", "tneb", "mseb", "uppcl", "dhbvn", "energy charges", "fixed charges"
        ],
        "weight": 2.0,
    },
    DocumentType.HEALTH_INSURANCE: {
        "keywords": [
            "health insurance", "mediclaim", "family floater", "hospitalization",
            "sum insured", "cumulative bonus", "pre-existing", "cashless", "tpa",
            "ayush", "day care procedures", "critical illness", "room rent",
            "medical expenses", "proposer", "covered members", "health shield"
        ],
        "weight": 1.8,
    },
    DocumentType.MOTOR_INSURANCE: {
        "keywords": [
            "motor insurance", "private car", "two wheeler", "commercial vehicle",
            "registration no", "chassis no", "engine no", "idv", "ncb",
            "own damage", "third party liability", "compulsory deductible",
            "seating capacity", "cubic capacity", "cpa cover", "puc", "rto",
            "certificate of insurance cum schedule", "package policy", "liability only"
        ],
        "weight": 1.8,
    },
    DocumentType.PROPERTY_INSURANCE: {
        "keywords": [
            "standard fire and special perils", "fire insurance", "property damage",
            "plant and machinery", "building and plinth", "stocks", "burglary",
            "earthquake", "terrorism", "risk location", "occupancy", "hazard"
        ],
        "weight": 1.8,
    },
    DocumentType.WORKMEN_COMPENSATION: {
        "keywords": [
            "workmen compensation", "employees compensation", "workmen's compensation",
            "w.c. act", "fatal accidents act", "nature of work", "estimated wages",
            "clerical", "manual labor", "occupational disease", "table a"
        ],
        "weight": 2.0,
    },
    DocumentType.INVOICE: {
        "keywords": [
            "tax invoice", "bill to", "ship to", "gstin", "invoice date", "invoice no",
            "hsn/sac", "cgst", "sgst", "igst", "total taxable value", "grand total"
        ],
        "weight": 1.5,
    },
}


def detect_motor_subtype(text_lower: str) -> MotorSubtype:
    """Classify the specific motor vehicle category based on primary policy indicators."""
    if any(k in text_lower for k in ["commercial vehicle", "goods carrier", "public carrier", "tata ace", "gvw", "truck", "load body"]):
        return MotorSubtype.COMMERCIAL_VEHICLE
    if any(k in text_lower for k in ["tractor", "agricultural tractor", "trailor"]):
        return MotorSubtype.TRACTOR
    if any(k in text_lower for k in ["motor cycle", "motorcycle", "scooter", "moped", "two wheeler package", "two wheeler policy", "2-wheeler", "two wheeler"]):
        return MotorSubtype.TWO_WHEELER
    if any(k in text_lower for k in ["private car", "car package", "sedan", "hatchback", "suv"]):
        return MotorSubtype.CAR
    return MotorSubtype.CAR  # Default motor subtype


def classify_document_content(text: str) -> Tuple[DocumentType, Optional[str]]:
    """
    Deterministically classifies document based SOLELY on content, NEVER filename.
    Returns: (DocumentType, subtype_or_none)
    """
    if not text or not text.strip():
        logger.warning("Empty text received for classification; returning GENERIC_DOCUMENT")
        return DocumentType.GENERIC_DOCUMENT, None

    text_lower = text.lower()
    scores = {}

    for doc_type, rule in CLASSIFICATION_RULES.items():
        score = 0.0
        kw_list = rule["keywords"]
        weight = rule["weight"]
        for kw in kw_list:
            # Count occurrences with word boundaries where appropriate
            occurrences = text_lower.count(kw)
            if occurrences > 0:
                score += min(occurrences, 4) * weight

        scores[doc_type] = score

    # Find highest scoring type
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_type, best_score = sorted_scores[0]

    logger.info(f"Classification scores: {sorted_scores[:3]}")

    if best_score < 3.0:
        # Not confident enough in specific domains
        # Check if invoice or generic receipt
        if "invoice" in text_lower or "bill" in text_lower:
            return DocumentType.INVOICE, None
        return DocumentType.GENERIC_DOCUMENT, None

    subtype = None
    if best_type == DocumentType.MOTOR_INSURANCE:
        subtype = detect_motor_subtype(text_lower).value

    return best_type, subtype
