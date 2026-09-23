"""
Extraction Router: Decides optimal extraction engines per page and merges results.
Avoids unnecessary OCR on native text pages while triggering OCR on scanned pages.
"""

from pathlib import Path
from typing import Dict, Any, List
from schemas.base import (
    DocumentInspectionResult,
    PageInspectionResult,
    PageType,
)
from ingestion.text_extractor import extract_native_text
from ingestion.ocr_engine import ocr_page
from ingestion.table_extractor import extract_tables_from_page
from ingestion.form_extractor import extract_form_fields
from utils.logger import get_logger

logger = get_logger("extraction_router")


def route_and_extract(
    file_path: Path,
    inspection: DocumentInspectionResult,
) -> Dict[str, Any]:
    """
    Executes routed extraction across all pages of the document.
    Returns:
    {
        "full_text": str,
        "page_texts": {page_num: str},
        "page_sources": {page_num: "native" | "ocr" | "hybrid"},
        "tables": List[Dict],
        "form_fields": Dict[str, Any],
        "ocr_invoked_pages": List[int],
        "extraction_errors": List[str],
    }
    """
    page_texts: Dict[int, str] = {}
    page_sources: Dict[int, str] = {}
    all_tables: List[Dict[str, Any]] = []
    ocr_invoked_pages: List[int] = []
    extraction_errors: List[str] = []

    # 1. First extract AcroForm fields if document has forms
    form_fields = extract_form_fields(file_path)

    # 2. Process each page according to its classified PageType
    for page_res in inspection.pages:
        pno = page_res.page_number
        ptype = page_res.page_type

        # Strategy A: Pure Scanned Image or requires OCR
        if ptype == PageType.SCANNED_IMAGE or page_res.requires_ocr:
            logger.info(f"Page {pno} requires OCR (Type: {ptype.value})")
            ocr_res = ocr_page(file_path, pno)
            if ocr_res["success"] and ocr_res["text"]:
                page_texts[pno] = ocr_res["text"]
                page_sources[pno] = "ocr"
                ocr_invoked_pages.append(pno)
            else:
                if ocr_res.get("error"):
                    extraction_errors.append(f"Page {pno} OCR: {ocr_res['error']}")
                # Fallback to any sparse native text
                native_dict = extract_native_text(file_path, [pno])
                fallback_txt = native_dict.get(pno, {}).get("text", "")
                page_texts[pno] = fallback_txt
                page_sources[pno] = "native_fallback"

        # Strategy B: Table-heavy page
        elif ptype == PageType.TABLE:
            logger.info(f"Page {pno} routed to Table Extractor + Text Extractor")
            # Extract tables
            tabs = extract_tables_from_page(file_path, pno)
            for t in tabs:
                t["page_number"] = pno
            all_tables.extend(tabs)

            # Also extract surrounding text
            native_dict = extract_native_text(file_path, [pno])
            page_texts[pno] = native_dict.get(pno, {}).get("text", "")
            page_sources[pno] = "native_table"

        # Strategy C: Hybrid or Text and Image
        elif ptype in (PageType.TEXT_AND_IMAGE, PageType.MIXED):
            logger.info(f"Page {pno} routed to Hybrid Extractor")
            native_dict = extract_native_text(file_path, [pno])
            ntxt = native_dict.get(pno, {}).get("text", "")

            # If text is sparse despite large image coverage, run supplementary OCR
            if len(ntxt) < 200 and page_res.image_coverage_ratio > 0.4:
                ocr_res = ocr_page(file_path, pno)
                if ocr_res["success"] and len(ocr_res["text"]) > len(ntxt):
                    page_texts[pno] = f"{ntxt}\n{ocr_res['text']}".strip()
                    page_sources[pno] = "hybrid"
                    ocr_invoked_pages.append(pno)
                else:
                    page_texts[pno] = ntxt
                    page_sources[pno] = "native"
            else:
                page_texts[pno] = ntxt
                page_sources[pno] = "native"

            # Check tables on hybrid page as well
            if page_res.likely_table:
                tabs = extract_tables_from_page(file_path, pno)
                for t in tabs:
                    t["page_number"] = pno
                all_tables.extend(tabs)

        # Strategy D: Native Text / Form / Standard
        else:
            logger.info(f"Page {pno} routed to Native Text Extractor")
            native_dict = extract_native_text(file_path, [pno])
            page_texts[pno] = native_dict.get(pno, {}).get("text", "")
            page_sources[pno] = "native"

            # Always extract structured tables if present or likely
            if page_res.likely_table or ptype == PageType.FORM:
                tabs = extract_tables_from_page(file_path, pno)
                for t in tabs:
                    t["page_number"] = pno
                all_tables.extend(tabs)

    from ingestion.document_layout import DocumentLayoutModel

    doc_id = getattr(inspection, "document_id", file_path.stem)
    layout = DocumentLayoutModel(document_id=doc_id)
    layout.add_tables(all_tables)

    # Populate layout model with page elements if available
    for pno in inspection.pages:
        p_num = pno.page_number
        native_data = extract_native_text(file_path, [p_num]).get(p_num, {})
        tokens = native_data.get("tokens", [])
        lines = native_data.get("lines", [])
        blocks = native_data.get("blocks", [])
        layout.add_page_elements(page=p_num, tokens=tokens, lines=lines, blocks=blocks)

    # Assemble full document text in natural reading order
    ordered_pages = sorted(page_texts.keys())
    full_text_parts = []
    for pno in ordered_pages:
        txt = page_texts.get(pno, "").strip()
        if txt:
            full_text_parts.append(f"--- Page {pno} ---\n{txt}")

    full_text = "\n\n".join(full_text_parts)

    return {
        "full_text": full_text,
        "page_texts": page_texts,
        "page_sources": page_sources,
        "tables": all_tables,
        "form_fields": form_fields,
        "ocr_invoked_pages": ocr_invoked_pages,
        "extraction_errors": extraction_errors,
        "layout": layout,
    }
