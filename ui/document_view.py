"""
Document View UI: Dynamic presentation of validated canonical document intelligence.
Sections 28, 51, 52, 54, 55: Domain-tailored cards, clean fields, export actions.
"""

import json
from typing import Optional, Any
import streamlit as st
import pandas as pd
from schemas.base import CanonicalDocument
from ui.chat import render_document_chat


def clean_canonical_data(data: Any) -> Any:
    """
    Ensure the canonical data is JSON-serializable and contains only primitive values:
    string, number, boolean, null, arrays, and objects.

    Do NOT expose internal candidate objects such as FieldCandidate or ResolvedFieldCandidate.
    If an internal candidate object exists, convert/use its final `.value` before displaying it.
    Never use `default=str` to serialize internal candidate objects.
    """
    if data is None:
        return None

    # Handle internal candidate and evidence objects (FieldCandidate, ResolvedFieldCandidate, SourceEvidence, Enum)
    if hasattr(data, "value") and (
        hasattr(data, "raw_evidence")
        or hasattr(data, "method")
        or "candidate" in type(data).__name__.lower()
        or "evidence" in type(data).__name__.lower()
    ):
        return clean_canonical_data(data.value)

    # Handle Pydantic models (v2 model_dump or v1 dict)
    if hasattr(data, "model_dump") and callable(getattr(data, "model_dump")):
        return clean_canonical_data(data.model_dump())
    if hasattr(data, "dict") and callable(getattr(data, "dict")):
        return clean_canonical_data(data.dict())

    # Handle generic objects with .value attribute (e.g. Enums)
    if hasattr(data, "value"):
        return clean_canonical_data(data.value)

    if isinstance(data, dict):
        clean_dict = {}
        for k, v in data.items():
            k_str = str(k)
            # Hide internal / private / candidate metadata keys
            if k_str.startswith("_") or "candidate" in k_str.lower():
                continue
            clean_dict[k_str] = clean_canonical_data(v)
        return clean_dict

    if isinstance(data, (list, tuple, set)):
        return [clean_canonical_data(item) for item in data]

    if isinstance(data, (str, int, float, bool)):
        return data

    if hasattr(data, "isoformat"):
        return data.isoformat()

    return str(data)


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

    # Clean canonical data object
    clean_data = clean_canonical_data(doc.structured_data)

    # Tab navigation for clean organization
    tab_overview, tab_chat, tab_export = st.tabs(["📋 Overview & Intelligence", "💬 Ask Document", "📥 Export Data"])

    with tab_overview:
        # Clean Normalized Canonical Data View
        with st.expander("View Normalized Canonical Data", expanded=False):
            st.caption("Clean canonical data representation used by enterprise systems")
            st.json(clean_data)

    with tab_chat:
        render_document_chat(doc)

    with tab_export:
        st.markdown("### 📥 Export Canonical Data")
        st.caption("Download clean, validated, normalized document data without technical noise.")

        export_col1, export_col2 = st.columns(2)

        # 1. JSON Export
        json_data = json.dumps(clean_data, indent=2)
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
        for k, v in clean_data.items():
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
