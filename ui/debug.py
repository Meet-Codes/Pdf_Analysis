"""
Debug UI: Developer diagnostic mode.
Section 44: Hidden from normal users. Reveals PDF inspection, OCR status,
raw extraction tokens, schema mapping, validation warnings, and source evidence.
"""

from typing import Optional
import streamlit as st
from schemas.base import CanonicalDocument
from config import settings


def render_developer_debug_panel(doc: Optional[CanonicalDocument]):
    """Renders developer diagnostics only when Debug Mode is enabled."""
    st.markdown("### 🛠️ Developer Diagnostics (Debug Mode)")

    if not doc:
        st.info("No document currently loaded.")
        return

    # System & Runtime Status
    with st.expander("⚙️ System & Engine Runtime", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("LLM Model", settings.OLLAMA_MODEL)
        c2.metric("Embedding Model", settings.EMBEDDING_MODEL)
        c3.metric("Tesseract Configured", "Yes" if settings.TESSERACT_CMD else "No")
        st.caption(f"Tesseract Path: `{settings.TESSERACT_CMD}`")
        st.caption(f"Ollama Endpoint: `{settings.OLLAMA_BASE_URL}`")

    # Inspection & Security
    if doc.inspection:
        with st.expander("📄 PDF Technical Inspection", expanded=True):
            insp = doc.inspection
            st.write(f"**Document ID**: `{insp.document_id}`")
            st.write(f"**File Name**: `{insp.file_name}` ({insp.file_size_bytes} bytes)")
            st.write(f"**Page Count**: {insp.page_count}")
            st.write(f"**Security Flags**: Encrypted={insp.is_encrypted}, Password={insp.requires_password}, CanCopy={insp.can_copy}, CanPrint={insp.can_print}")
            st.write(f"**Detected PDF Categories**: {', '.join(insp.detected_pdf_categories)}")
            st.write(f"**Requires OCR**: `{insp.overall_requires_ocr}`")

            # Page breakdown table
            page_data = []
            for p in insp.pages:
                page_data.append({
                    "Page": p.page_number,
                    "Type": p.page_type.value,
                    "Text Length": p.text_length,
                    "Images": p.image_count,
                    "Coverage": f"{p.image_coverage_ratio * 100:.1f}%",
                    "Requires OCR": p.requires_ocr,
                    "Has Signature": p.has_digital_signature,
                })
            st.dataframe(page_data, use_container_width=True)

    # Validation Engine Diagnostics
    with st.expander("⚖️ Validation & Reconciliation Log", expanded=False):
        st.write(f"**Status Valid**: `{doc.is_valid}`")
        if doc.validation_errors:
            st.error(f"Validation Errors ({len(doc.validation_errors)}):")
            for err in doc.validation_errors:
                st.write(f"- {err}")
        else:
            st.success("No blocking validation errors.")

        if doc.validation_warnings:
            st.warning(f"Validation Warnings ({len(doc.validation_warnings)}):")
            for warn in doc.validation_warnings:
                st.write(f"- {warn}")
        else:
            st.info("No validation warnings.")

    # Canonical Structured JSON
    with st.expander("📦 Normalized Canonical JSON", expanded=False):
        st.json(doc.structured_data)

    # Source Traceability & Evidence Map (Section 62)
    with st.expander("🔍 Source Evidence Trace (Section 62)", expanded=False):
        evidence_records = []
        for field, ev in doc.evidence.items():
            evidence_records.append({
                "Field": field,
                "Value": str(ev.value),
                "Page": ev.page,
                "Source Engine": ev.source,
                "Evidence Snippet": ev.evidence,
            })
        st.dataframe(evidence_records, use_container_width=True)
