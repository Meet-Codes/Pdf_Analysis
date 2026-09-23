"""
LangGraph Workflow: Production-grade orchestration of the Document Intelligence pipeline.
Strongly typed state transitions separating inspection, extraction, normalization, validation, and canonical building.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

from schemas.base import (
    DocumentInspectionResult,
    DocumentType,
    CanonicalDocument,
    SourceEvidence,
)
from ingestion.pdf_inspector import inspect_pdf
from ingestion.page_analyzer import analyze_all_pages
from ingestion.extraction_router import route_and_extract
from agents.classifier import classify_document_content
from agents.extractor import extract_structured_fields
from agents.normalizer import normalize_document_content
from validation.validator import validate_document_data
from agents.validator import build_canonical_sections
from agents.summarizer import generate_document_summary
from retrieval.retriever import index_canonical_document
from utils.logger import get_logger

logger = get_logger("workflow")


class DocumentState(TypedDict):
    document_id: str
    file_path: str
    file_name: str
    inspection: Optional[DocumentInspectionResult]
    extracted_content: Dict[str, Any]
    document_type: DocumentType
    document_subtype: Optional[str]
    raw_extraction: Dict[str, Any]
    normalized_data: Dict[str, Any]
    evidence_map: Dict[str, SourceEvidence]
    is_valid: bool
    validation_warnings: List[str]
    validation_errors: List[str]
    canonical_document: Optional[CanonicalDocument]
    summary: str
    errors: List[str]
    processing_status: str


def node_inspect(state: DocumentState) -> Dict[str, Any]:
    """Step 1: Inspect PDF technical properties, permissions, and security."""
    path = Path(state["file_path"])
    inspection = inspect_pdf(path, state["document_id"])
    return {
        "inspection": inspection,
        "processing_status": "inspected",
    }


def node_analyze_pages(state: DocumentState) -> Dict[str, Any]:
    """Step 2: Classify pages independently."""
    inspection = state["inspection"]
    if inspection and inspection.pages:
        inspection.pages = analyze_all_pages(inspection.pages)
    return {
        "inspection": inspection,
        "processing_status": "pages_analyzed",
    }


def node_route_and_extract(state: DocumentState) -> Dict[str, Any]:
    """Step 3: Route extraction (Text, OCR, Tables, Forms)."""
    path = Path(state["file_path"])
    inspection = state["inspection"]
    extracted = route_and_extract(path, inspection)
    return {
        "extracted_content": extracted,
        "processing_status": "content_extracted",
    }


def node_classify(state: DocumentState) -> Dict[str, Any]:
    """Step 4: Classify document based on CONTENT, never filename."""
    full_text = state["extracted_content"].get("full_text", "")
    doc_type, subtype = classify_document_content(full_text)
    return {
        "document_type": doc_type,
        "document_subtype": subtype,
        "processing_status": "classified",
    }


def node_extract_structured(state: DocumentState) -> Dict[str, Any]:
    """Step 5: Extract structured schema fields."""
    full_text = state["extracted_content"].get("full_text", "")
    tables = state["extracted_content"].get("tables", [])
    form_fields = state["extracted_content"].get("form_fields", {})
    doc_type = state["document_type"]

    page_texts = state["extracted_content"].get("page_texts", {})
    layout = state["extracted_content"].get("layout", None)

    raw_extracted = extract_structured_fields(
        text=full_text,
        doc_type=doc_type,
        tables=tables,
        form_fields=form_fields,
        page_texts=page_texts,
        layout=layout,
    )
    return {
        "raw_extraction": raw_extracted,
        "processing_status": "fields_extracted",
    }


def node_normalize(state: DocumentState) -> Dict[str, Any]:
    """Step 6: Field-level deterministic and semantic normalization."""
    raw_data = state["raw_extraction"]
    doc_type = state["document_type"]
    page_texts = state["extracted_content"].get("page_texts", {})
    page_sources = state["extracted_content"].get("page_sources", {})

    normalized, evidence = normalize_document_content(
        raw_data=raw_data,
        doc_type=doc_type,
        page_texts=page_texts,
        page_sources=page_sources,
    )
    return {
        "normalized_data": normalized,
        "evidence_map": evidence,
        "processing_status": "normalized",
    }


def node_validate(state: DocumentState) -> Dict[str, Any]:
    """Step 7: Run cross-field validation & reconciliation."""
    data = state["normalized_data"]
    doc_type = state["document_type"]
    is_valid, warnings, errors = validate_document_data(data, doc_type)
    return {
        "is_valid": is_valid,
        "validation_warnings": warnings,
        "validation_errors": errors,
        "processing_status": "validated",
    }


def node_build_canonical(state: DocumentState) -> Dict[str, Any]:
    """Step 8: Construct the single CanonicalDocument object."""
    doc_type = state["document_type"]
    subtype = state["document_subtype"]
    norm_data = state["normalized_data"]

    sections = build_canonical_sections(norm_data, doc_type, subtype)
    summary = generate_document_summary(norm_data, doc_type, subtype)

    title = f"{subtype or doc_type.value.replace('_', ' ').title()}"

    canonical = CanonicalDocument(
        document_id=state["document_id"],
        file_name=state["file_name"],
        document_type=doc_type,
        document_subtype=subtype,
        title=title,
        summary=summary,
        structured_data=norm_data,
        sections=sections,
        evidence=state["evidence_map"],
        is_valid=state["is_valid"],
        validation_warnings=state["validation_warnings"],
        validation_errors=state["validation_errors"],
        inspection=state["inspection"],
        page_texts=state["extracted_content"].get("page_texts", {}),
        tables=state["extracted_content"].get("tables", []),
    )

    # Index into RAG vector store & BM25 with tables as first-class retrieval chunks
    page_texts = state["extracted_content"].get("page_texts", {})
    tables = state["extracted_content"].get("tables", [])
    index_canonical_document(canonical, page_texts, tables=tables)

    return {
        "canonical_document": canonical,
        "summary": summary,
        "processing_status": "completed",
    }


def create_processing_graph():
    """Builds and compiles the LangGraph state machine."""
    workflow = StateGraph(DocumentState)

    workflow.add_node("inspect", node_inspect)
    workflow.add_node("analyze_pages", node_analyze_pages)
    workflow.add_node("route_and_extract", node_route_and_extract)
    workflow.add_node("classify", node_classify)
    workflow.add_node("extract_structured", node_extract_structured)
    workflow.add_node("normalize", node_normalize)
    workflow.add_node("validate", node_validate)
    workflow.add_node("build_canonical", node_build_canonical)

    workflow.set_entry_point("inspect")
    workflow.add_edge("inspect", "analyze_pages")
    workflow.add_edge("analyze_pages", "route_and_extract")
    workflow.add_edge("route_and_extract", "classify")
    workflow.add_edge("classify", "extract_structured")
    workflow.add_edge("extract_structured", "normalize")
    workflow.add_edge("normalize", "validate")
    workflow.add_edge("validate", "build_canonical")
    workflow.add_edge("build_canonical", END)

    return workflow.compile()


# Compiled LangGraph pipeline
processing_pipeline = create_processing_graph()


def process_document(file_path: Path, document_id: str, original_filename: str) -> CanonicalDocument:
    """
    Executes end-to-end document intelligence pipeline on a file.
    Returns the single CanonicalDocument result.
    """
    initial_state: DocumentState = {
        "document_id": document_id,
        "file_path": str(file_path),
        "file_name": original_filename,
        "inspection": None,
        "extracted_content": {},
        "document_type": DocumentType.GENERIC_DOCUMENT,
        "document_subtype": None,
        "raw_extraction": {},
        "normalized_data": {},
        "evidence_map": {},
        "is_valid": True,
        "validation_warnings": [],
        "validation_errors": [],
        "canonical_document": None,
        "summary": "",
        "errors": [],
        "processing_status": "started",
    }

    final_state = processing_pipeline.invoke(initial_state)
    canonical = final_state.get("canonical_document")
    if not canonical:
        raise RuntimeError("Workflow completed without producing a CanonicalDocument.")

    return canonical
