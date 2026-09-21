"""
Normalizer Agent: Normalizes raw extracted fields into clean canonical representation.
Builds field-level source evidence mappings for internal traceability.
"""

from typing import Dict, Any, Tuple
from schemas.base import DocumentType, SourceEvidence
from validation.normalizer import normalize_fields


def normalize_document_content(
    raw_data: Dict[str, Any],
    doc_type: DocumentType,
    page_texts: Dict[int, str] = None,
    page_sources: Dict[int, str] = None,
) -> Tuple[Dict[str, Any], Dict[str, SourceEvidence]]:
    """
    Normalizes all extracted fields and maps source evidence.
    Returns: (normalized_data, evidence_map)
    """
    normalized_data = normalize_fields(raw_data, doc_type)
    evidence_map: Dict[str, SourceEvidence] = {}

    # Build evidence map
    for field, val in normalized_data.items():
        if val is None:
            continue

        # Find which page contained the value
        target_val_str = str(val).lower()
        found_page = 1
        found_source = "text"
        found_line = ""

        if page_texts:
            for pno, txt in page_texts.items():
                if target_val_str in txt.lower() or str(raw_data.get(field, "")).lower() in txt.lower():
                    found_page = pno
                    found_source = page_sources.get(pno, "text") if page_sources else "text"
                    # Grab matching line as snippet
                    for line in txt.split("\n"):
                        if target_val_str in line.lower():
                            found_line = line.strip()
                            break
                    break

        evidence_map[field] = SourceEvidence(
            field=field,
            value=val,
            page=found_page,
            source=found_source,
            evidence=found_line or f"Extracted from page {found_page}",
        )

    return normalized_data, evidence_map
