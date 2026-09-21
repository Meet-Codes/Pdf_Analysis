"""
PDF Inspector: Technical inspection and page-by-page content profiling using PyMuPDF.
Determines encryption, digital signatures, AcroForms, embedded files, and page content types.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import fitz  # PyMuPDF
from schemas.base import (
    DocumentInspectionResult,
    PageInspectionResult,
    PageType,
)
from utils.logger import get_logger
from utils.file_utils import generate_document_id

logger = get_logger("pdf_inspector")


def inspect_pdf(file_path: Path, document_id: Optional[str] = None) -> DocumentInspectionResult:
    """
    Performs comprehensive technical inspection of a PDF file.
    Analyzes each page independently without loading all page bitmaps simultaneously.
    """
    if not document_id:
        document_id = generate_document_id(file_path.name)

    file_size = file_path.stat().st_size if file_path.exists() else 0

    doc = None
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        logger.error(f"Failed to open PDF {file_path}: {e}")
        return DocumentInspectionResult(
            document_id=document_id,
            file_name=file_path.name,
            file_size_bytes=file_size,
            page_count=0,
            detected_pdf_categories=["Corrupted/Malformed PDF"],
        )

    # Security & Password
    is_encrypted = doc.is_encrypted
    requires_password = doc.needs_pass
    perms = doc.permissions
    can_print = bool(perms & fitz.PDF_PERM_PRINT)
    can_modify = bool(perms & fitz.PDF_PERM_MODIFY)
    can_copy = bool(perms & fitz.PDF_PERM_COPY)
    can_annotate = bool(perms & fitz.PDF_PERM_ANNOTATE)

    if requires_password:
        doc.close()
        return DocumentInspectionResult(
            document_id=document_id,
            file_name=file_path.name,
            file_size_bytes=file_size,
            page_count=0,
            is_encrypted=is_encrypted,
            requires_password=True,
            can_copy=can_copy,
            can_print=can_print,
            can_modify=can_modify,
            can_annotate=can_annotate,
            detected_pdf_categories=["Password-protected PDF"],
        )

    # Embedded files / Attachments
    embedded_names = []
    try:
        embedded_names = doc.embfile_names()
    except Exception:
        pass
    has_embedded_files = len(embedded_names) > 0

    # Digital signatures
    has_digital_signatures = False
    try:
        # Check sigflags or signature fields in PDF
        # PyMuPDF has doc.get_sigflags()
        sig_flags = doc.get_sigflags()
        if sig_flags > 0:
            has_digital_signatures = True
    except Exception:
        pass

    page_count = len(doc)
    page_results: List[PageInspectionResult] = []
    overall_requires_ocr = False
    detected_categories = set()

    if is_encrypted:
        detected_categories.add("Encrypted PDF")
    if not can_copy:
        detected_categories.add("Permission-restricted PDF")
    if has_embedded_files:
        detected_categories.add("PDF with embedded attachments")
    if has_digital_signatures:
        detected_categories.add("Digitally signed PDF")

    # Analyze each page independently
    for pno in range(page_count):
        page = doc[pno]
        rect = page.rect
        page_area = rect.width * rect.height if rect.width > 0 and rect.height > 0 else 1.0

        # Extract native text
        text = page.get_text("text").strip()
        text_length = len(text)
        has_text = text_length > 30

        # Extract images info safely
        images = []
        try:
            images = page.get_images(full=True)
        except Exception as e:
            logger.warning(f"Could not read images on page {pno + 1}: {e}")
        image_count = len(images)

        # Estimate image coverage area
        image_coverage_area = 0.0
        for img_info in images:
            xref = img_info[0]
            try:
                img_rects = page.get_image_rects(xref)
                for ir in img_rects:
                    image_coverage_area += ir.width * ir.height
            except Exception:
                pass

        coverage_ratio = min(1.0, image_coverage_area / page_area) if page_area > 0 else 0.0

        # Check for tables
        likely_table = False
        try:
            # fitz has page.find_tables() in modern PyMuPDF
            tabs = page.find_tables()
            if tabs.tables and len(tabs.tables) > 0:
                likely_table = True
        except Exception:
            # Fallback heuristic: lines or grid-like layout
            if "table" in text.lower() or "\t" in text or text.count("\n") > 20:
                likely_table = True

        # Check for AcroForm fields
        likely_form = False
        try:
            widgets = list(page.widgets())
            if len(widgets) > 0:
                likely_form = True
                detected_categories.add("Fillable form PDF")
        except Exception:
            pass

        # Determine if page is scanned or requires OCR
        # Scanned page: low text length (<50 chars) and has images with significant coverage (>30%)
        # or zero text with image
        likely_scanned = (text_length < 50 and (image_count > 0 or coverage_ratio > 0.3)) or (text_length == 0 and image_count > 0)
        requires_ocr = likely_scanned or (coverage_ratio > 0.5 and text_length < 200)

        if requires_ocr:
            overall_requires_ocr = True

        # Determine PageType
        if likely_form:
            page_type = PageType.FORM
        elif likely_scanned:
            page_type = PageType.SCANNED_IMAGE
        elif likely_table and has_text:
            page_type = PageType.TABLE
        elif has_text and image_count > 0:
            page_type = PageType.TEXT_AND_IMAGE
        elif has_text:
            page_type = PageType.TEXT
        elif image_count > 0:
            page_type = PageType.IMAGE
        else:
            page_type = PageType.UNKNOWN

        page_inspection = PageInspectionResult(
            page_number=pno + 1,
            has_text=has_text,
            text_length=text_length,
            image_count=image_count,
            image_coverage_ratio=round(coverage_ratio, 3),
            likely_scanned=likely_scanned,
            likely_table=likely_table,
            likely_form=likely_form,
            has_digital_signature=has_digital_signatures,
            requires_ocr=requires_ocr,
            page_type=page_type,
            width=rect.width,
            height=rect.height,
        )
        page_results.append(page_inspection)

    # Document-level categories
    scanned_pages = sum(1 for p in page_results if p.likely_scanned)
    text_pages = sum(1 for p in page_results if p.has_text and not p.likely_scanned)

    if scanned_pages == page_count and page_count > 0:
        detected_categories.add("Scanned PDF")
    elif text_pages == page_count and page_count > 0:
        detected_categories.add("Native text PDF")
    elif scanned_pages > 0 and text_pages > 0:
        detected_categories.add("Hybrid PDF")
        detected_categories.add("Mixed-content PDF")
    else:
        detected_categories.add("Native text PDF")

    if any(p.likely_table for p in page_results):
        detected_categories.add("Table-heavy PDF")

    metadata = doc.metadata or {}
    doc.close()

    return DocumentInspectionResult(
        document_id=document_id,
        file_name=file_path.name,
        file_size_bytes=file_size,
        page_count=page_count,
        is_encrypted=is_encrypted,
        requires_password=requires_password,
        can_copy=can_copy,
        can_print=can_print,
        can_modify=can_modify,
        can_annotate=can_annotate,
        has_embedded_files=has_embedded_files,
        embedded_file_names=embedded_names,
        has_digital_signatures=has_digital_signatures,
        detected_pdf_categories=sorted(list(detected_categories)),
        overall_requires_ocr=overall_requires_ocr,
        pages=page_results,
        metadata=metadata,
    )
