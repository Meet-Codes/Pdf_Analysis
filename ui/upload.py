"""
Upload UI: Professional drag-and-drop file ingestion interface.
Section 6: Clean pipeline status checkmarks without technical noise.
"""

from typing import List, Any
import streamlit as st


def render_upload_zone() -> List[Any]:
    """
    Renders the commercial-grade upload area.
    Returns list of uploaded files.
    """
    st.markdown(
        """
        <div class="upload-zone">
            <h2 style="margin: 0 0 0.5rem 0; font-size: 1.5rem; font-weight: 700; color: #F8FAFC;">DOCUMENT INTELLIGENCE</h2>
            <p style="margin: 0 0 1.5rem 0; color: #94A3B8; font-size: 0.95rem;">Understand your documents automatically with enterprise precision.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        label="Drag & Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload single or multiple PDF documents for automatic intelligence extraction.",
        label_visibility="collapsed",
    )

    st.caption("Supported: PDF documents (Native, Scanned, Forms, Tables, Hybrid)")
    return uploaded_files or []


def render_processing_progress():
    """Renders the clean, customer-facing verification progress steps."""
    steps_html = """
    <div style="background: #1E293B; border-radius: 8px; padding: 1rem; border: 1px solid rgba(255,255,255,0.06); margin: 1rem 0;">
        <div class="step-item">✓ File uploaded</div>
        <div class="step-item">✓ Document inspected</div>
        <div class="step-item">✓ Content analyzed</div>
        <div class="step-item">✓ Extraction completed</div>
        <div class="step-item">✓ Information validated</div>
        <div class="step-item">✓ Document ready</div>
    </div>
    """
    st.markdown(steps_html, unsafe_allow_html=True)
