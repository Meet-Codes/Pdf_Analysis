"""
Summary UI: Renders the executive summary of the active document.
"""

import streamlit as st


def render_summary_card(summary_text: str, title: str = "Executive Summary"):
    """Renders the clean document summary card."""
    if not summary_text:
        return

    with st.container():
        st.markdown(
            f"""
            <div class="card-header">
                <span>📋</span> <span>{title}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(summary_text)
