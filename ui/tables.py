"""
Tables UI: Clean tabular presentation of complex multi-row extracted content.
"""

from typing import List, Dict, Any
import streamlit as st
import pandas as pd


def render_extracted_tables(tables: List[Dict[str, Any]]):
    """Renders structured tables detected and parsed from the document."""
    if not tables:
        return

    st.markdown(
        """
        <div class="card-header">
            <span>📊</span> <span>Extracted Tables</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for idx, table in enumerate(tables):
        pno = table.get("page_number", 1)
        headers = table.get("headers", [])
        rows = table.get("rows", [])

        if headers and rows:
            try:
                # Pad rows to match headers length
                clean_rows = []
                for r in rows:
                    if len(r) < len(headers):
                        r = r + [""] * (len(headers) - len(r))
                    clean_rows.append(r[:len(headers)])

                df = pd.DataFrame(clean_rows, columns=headers)
                st.caption(f"Table {idx + 1} (Page {pno})")
                st.dataframe(df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"Could not render table {idx + 1}: {e}")
