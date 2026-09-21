"""
UI Cards: Reusable rendering components for clean production display.
Section 52 & 53: Clean field display, zero raw OCR garbage, zero None/null values.
"""

from typing import Dict, Any, List
import streamlit as st


def render_field(label: str, value: Any, is_highlight: bool = False):
    """
    Renders a single label-value field cleanly.
    Hides internal technical details; formats missing values cleanly.
    """
    val_str = str(value).strip() if value is not None else "Not available in the document"
    is_missing = val_str == "Not available in the document"

    if is_missing:
        val_html = f'<div class="field-value field-missing">{val_str}</div>'
    elif is_highlight:
        val_html = f'<div class="field-value field-highlight">{val_str}</div>'
    else:
        val_html = f'<div class="field-value">{val_str}</div>'

    st.markdown(
        f"""
        <div class="field-pair">
            <div class="field-label">{label}</div>
            {val_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_card(section: Dict[str, Any], columns: int = 2):
    """
    Renders a grouped section card containing multiple fields in a responsive grid.
    """
    title = section.get("title", "Details")
    icon = section.get("icon", "📋")
    fields: List[Dict[str, Any]] = section.get("fields", [])

    st.markdown(
        f"""
        <div class="card-header">
            <span>{icon}</span> <span>{title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Distribute fields across columns
    cols = st.columns(columns)
    for idx, field in enumerate(fields):
        with cols[idx % columns]:
            render_field(
                label=field.get("label", ""),
                value=field.get("value"),
                is_highlight=field.get("is_highlight", False),
            )
