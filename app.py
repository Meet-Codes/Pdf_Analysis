"""
Production Document Intelligence Platform - Streamlit Application Entry Point.
Commercial-grade architecture separating inspection, extraction, normalization, validation, and presentation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import streamlit as st

# Set page configuration FIRST before any other Streamlit calls
st.set_page_config(
    page_title="Document Intelligence Platform",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import settings
from utils.logger import get_logger
from utils.file_utils import save_uploaded_file, generate_document_id
from utils.security import check_pdf_security
from schemas.base import CanonicalDocument
from agents.workflow import process_document
from ui.layout import inject_custom_css
from ui.upload import render_upload_zone, render_processing_progress
from ui.document_view import render_document_view
from ui.debug import render_developer_debug_panel

logger = get_logger("app")


def initialize_session_state():
    """Initialize persistent session states."""
    if "documents" not in st.session_state:
        st.session_state["documents"] = {}  # doc_id -> CanonicalDocument
    if "active_doc_id" not in st.session_state:
        st.session_state["active_doc_id"] = None
    if "debug_mode" not in st.session_state:
        st.session_state["debug_mode"] = settings.ENABLE_DEBUG_MODE
    if "recent_docs" not in st.session_state:
        st.session_state["recent_docs"] = []


def render_sidebar():
    """Renders the standard enterprise sidebar navigation and settings."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.5rem 0 1rem 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 1rem;">
                <h3 style="margin: 0; font-weight: 700; color: #F8FAFC; font-size: 1.2rem;">DocIntelligence</h3>
                <span style="font-size: 0.75rem; color: #10B981; font-weight: 600;">● SYSTEM OPERATIONAL</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("DOCUMENTS")
        nav_mode = st.radio(
            "Navigation",
            options=["Upload Documents", "My Documents", "Recent Documents"],
            label_visibility="collapsed",
        )

        # Document switcher if documents are loaded
        if st.session_state["documents"]:
            st.caption("Active Document:")
            doc_options = {doc_id: doc.file_name for doc_id, doc in st.session_state["documents"].items()}
            current_active = st.session_state["active_doc_id"]
            selected_id = st.selectbox(
                "Select Document",
                options=list(doc_options.keys()),
                format_func=lambda x: doc_options[x],
                index=list(doc_options.keys()).index(current_active) if current_active in doc_options else 0,
                label_visibility="collapsed",
            )
            st.session_state["active_doc_id"] = selected_id

        st.markdown("---")
        st.subheader("SETTINGS")
        st.caption(f"LLM Engine: `{settings.OLLAMA_MODEL}`")
        st.caption(f"OCR: `{'Available' if settings.TESSERACT_CMD else 'Unavailable'}`")

        st.markdown("---")
        st.subheader("DEVELOPER")
        st.session_state["debug_mode"] = st.toggle("Enable Debug Mode", value=st.session_state["debug_mode"])
        if st.session_state["debug_mode"]:
            st.caption("🛠️ Diagnostics enabled")

    return nav_mode


import hashlib

PIPELINE_VERSION = "2.1.0"


def handle_file_processing(uploaded_files: List[Any]):
    """Processes uploaded files through the complete intelligence pipeline."""
    if not uploaded_files:
        return

    for uploaded_file in uploaded_files:
        fname = uploaded_file.name
        file_bytes = uploaded_file.read()

        # Cache key based on SHA-256(file_bytes) + pipeline_version
        file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
        doc_id = f"{file_hash}_{PIPELINE_VERSION}"

        if doc_id in st.session_state["documents"]:
            st.session_state["active_doc_id"] = doc_id
            continue

        with st.status(f"Processing {fname}...", expanded=True) as status:
            st.write("Verifying document security and permissions...")
            temp_path = save_uploaded_file(file_bytes, fname)

            # Security inspection
            sec_info = check_pdf_security(temp_path)
            if sec_info.get("requires_password"):
                status.update(label=f"Password Required for {fname}", state="error")
                st.error("This document is password protected. Please provide an authorized unlocked copy or password.")
                continue

            if not sec_info.get("is_valid_pdf"):
                status.update(label=f"Invalid Document {fname}", state="error")
                st.error(sec_info.get("error_message") or "Unable to process this document.")
                continue

            st.write("Inspecting structure and classifying content...")
            try:
                canonical = process_document(temp_path, doc_id, fname)
                st.session_state["documents"][doc_id] = canonical
                st.session_state["active_doc_id"] = doc_id
                if fname not in st.session_state["recent_docs"]:
                    st.session_state["recent_docs"].append(fname)

                render_processing_progress()
                status.update(label=f"✓ {fname} Processed Successfully", state="complete")
            except Exception as e:
                logger.error(f"Pipeline error on {fname}: {e}", exc_info=True)
                status.update(label=f"Processing Issue on {fname}", state="error")
                st.error("Unable to process this document.")
                if st.session_state["debug_mode"]:
                    st.exception(e)


def main():
    inject_custom_css()
    initialize_session_state()

    # Top App Header
    st.markdown(
        """
        <div class="app-header">
            <div>
                <h1 class="app-title">Document Intelligence Platform</h1>
                <div class="app-subtitle">Automated Inspection, Extraction, Normalization & Grounded Q&A</div>
            </div>
            <div>
                <span class="badge-primary">PRODUCTION READY</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_mode = render_sidebar()

    # 1. Upload View
    if nav_mode == "Upload Documents":
        uploaded_files = render_upload_zone()
        if uploaded_files:
            handle_file_processing(uploaded_files)

        # Show active document if available
        active_id = st.session_state.get("active_doc_id")
        if active_id and active_id in st.session_state["documents"]:
            active_doc = st.session_state["documents"][active_id]
            st.markdown("---")
            render_document_view(active_doc)

            # Debug mode panel at the bottom if enabled
            if st.session_state["debug_mode"]:
                st.markdown("---")
                render_developer_debug_panel(active_doc)

    # 2. My Documents View
    elif nav_mode == "My Documents":
        if not st.session_state["documents"]:
            st.info("No documents uploaded yet. Go to 'Upload Documents' to get started.")
        else:
            active_id = st.session_state.get("active_doc_id")
            if active_id and active_id in st.session_state["documents"]:
                active_doc = st.session_state["documents"][active_id]
                render_document_view(active_doc)
                if st.session_state["debug_mode"]:
                    st.markdown("---")
                    render_developer_debug_panel(active_doc)

    # 3. Recent Documents View
    elif nav_mode == "Recent Documents":
        st.subheader("Recent Ingestion History")
        if not st.session_state["recent_docs"]:
            st.caption("No recent documents recorded.")
        else:
            for item in st.session_state["recent_docs"]:
                st.markdown(f"- 📄 `{item}`")


if __name__ == "__main__":
    main()
