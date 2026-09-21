"""
Page Analyzer: Classifies each page independently into deterministic page types.
Used by the Extraction Router to select the optimal parsing strategy per page.
"""

from typing import List
from schemas.base import PageInspectionResult, PageType


def analyze_page_type(page_inspection: PageInspectionResult) -> PageType:
    """
    Deterministically computes the PageType for a given page inspection result.
    Prevents unnecessary OCR on native text pages while ensuring scanned pages get OCR.
    """
    if page_inspection.likely_form:
        return PageType.FORM

    if page_inspection.likely_scanned:
        return PageType.SCANNED_IMAGE

    if page_inspection.likely_table and page_inspection.has_text:
        return PageType.TABLE

    if page_inspection.has_text and page_inspection.image_count > 0:
        return PageType.TEXT_AND_IMAGE

    if page_inspection.has_text:
        return PageType.TEXT

    if page_inspection.image_count > 0:
        return PageType.IMAGE

    if page_inspection.has_digital_signature:
        return PageType.SIGNATURE

    return PageType.UNKNOWN


def analyze_all_pages(pages: List[PageInspectionResult]) -> List[PageInspectionResult]:
    """Analyze and update page types for all pages."""
    for p in pages:
        p.page_type = analyze_page_type(p)
    return pages
