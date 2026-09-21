"""
Document View UI: Dynamic presentation of validated canonical document intelligence.
Sections 28, 51, 52, 54, 55: Domain-tailored cards, clean fields, export actions.
"""

import json
from typing import Optional
import streamlit as st
import pandas as pd
from schemas.base import CanonicalDocument
from ui.cards import render_section_card
from ui.summary import render_summary_card
from ui.chat import render_document_chat


def render_document_view(doc: CanonicalDocument):
    """Renders the comprehensive, production-grade document intelligence view."""
    # Top Header
    badge_class = "badge-success" if doc.is_valid else "badge-warning"
    status_label = "Validated Document" if doc.is_valid else "Validated with Warnings"

    st.markdown(
        f"""
        <div style="display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 1.5rem;">
            <div>
                <h2 style="margin: 0; font-size: 1.8rem; font-weight: 700; color: #F8FAFC;">{doc.title}</h2>
                <div style="margin-top: 0.35rem;">
                    <span class="badge-primary">{doc.document_type.value}</span>
                    <span class="{badge_class}" style="margin-left: 0.5rem;">{status_label}</span>
                </div>
            </div>
            <div style="font-size: 0.85rem; color: #94A3B8;">
                File: <code>{doc.file_name}</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Validation warnings banner if applicable
    if doc.validation_warnings:
        with st.container():
            st.warning("⚠️ Attention: " + "; ".join(doc.validation_warnings))

    # Tab navigation for clean organization
    tab_overview, tab_chat, tab_export = st.tabs(["📋 Overview & Intelligence", "💬 Ask Document", "📥 Export Data"])

    with tab_overview:
        # Executive Summary
        if doc.summary:
            with st.container():
                st.markdown('<div class="doc-card">', unsafe_allow_html=True)
                render_summary_card(doc.summary)
                st.markdown('</div>', unsafe_allow_html=True)

        # Dynamic Section Cards (Customer, Policy, Vehicle, Financials, Dates, etc.)
        for section in doc.sections:
            st.markdown('<div class="doc-card">', unsafe_allow_html=True)
            render_section_card(section, columns=2)
            st.markdown('</div>', unsafe_allow_html=True)

        # Clean Structured Data View (Section 54)
        with st.expander("🔍 View Normalized Canonical Data", expanded=False):
            st.caption("Clean canonical data representation used by enterprise systems:")
            st.json(doc.structured_data)

    with tab_chat:
        render_document_chat(doc)

    with tab_export:
        st.markdown("### 📥 Export Canonical Data")
        st.caption("Download clean, validated, normalized document data without technical noise.")

        export_col1, export_col2 = st.columns(2)

        # 1. JSON Export
        json_data = json.dumps(doc.structured_data, indent=2)
        clean_name = doc.file_name.rsplit(".", 1)[0]
        export_col1.download_button(
            label="📄 Download JSON",
            data=json_data,
            file_name=f"{clean_name}_canonical.json",
            mime="application/json",
            use_container_width=True,
        )

        # 2. CSV Export
        flat_records = []
        for k, v in doc.structured_data.items():
            if not isinstance(v, (dict, list)):
                flat_records.append({"Field": k, "Normalized Value": v})
        csv_df = pd.DataFrame(flat_records)
        csv_data = csv_df.to_csv(index=False)
        export_col2.download_button(
            label="📊 Download CSV",
            data=csv_data,
            file_name=f"{clean_name}_canonical.csv",
            mime="text/csv",
            use_container_width=True,
        )
