"""
UI Layout and Styling: Injects sleek, commercial SaaS design into Streamlit.
Harmonious color palettes, subtle cards, responsive grid, and modern typography.
"""

import streamlit as st


def inject_custom_css():
    """Injects production-grade custom CSS styling."""
    custom_css = """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Header Bar */
    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.25rem 0 1.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 2rem;
    }

    .app-title {
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #F8FAFC;
        margin: 0;
    }

    .app-subtitle {
        font-size: 0.88rem;
        color: #94A3B8;
        margin-top: 0.2rem;
    }

    /* Badges */
    .badge-primary {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    .badge-success {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .badge-warning {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* Cards */
    .doc-card {
        background: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    .card-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.05rem;
        font-weight: 600;
        color: #F1F5F9;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        padding-bottom: 0.5rem;
    }

    /* Field Pair */
    .field-pair {
        margin-bottom: 0.85rem;
    }

    .field-label {
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 0.2rem;
    }

    .field-value {
        font-size: 0.95rem;
        font-weight: 500;
        color: #F8FAFC;
        word-break: break-word;
    }

    .field-highlight {
        font-size: 1.15rem;
        font-weight: 700;
        color: #38BDF8;
    }

    .field-missing {
        color: #64748B;
        font-style: italic;
        font-size: 0.85rem;
    }

    /* Upload Zone */
    .upload-zone {
        border: 2px dashed rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 2.5rem 1.5rem;
        text-align: center;
        background: rgba(30, 41, 59, 0.5);
        margin-bottom: 1.5rem;
    }

    /* Pipeline Step Checkmarks */
    .step-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.85rem;
        color: #10B981;
        margin-bottom: 0.35rem;
    }

    /* Chat bubble */
    .chat-user {
        background: #2563EB;
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 12px 12px 0 12px;
        margin-bottom: 0.5rem;
        max-width: 80%;
        margin-left: auto;
    }

    .chat-assistant {
        background: #334155;
        color: #F8FAFC;
        padding: 0.75rem 1rem;
        border-radius: 12px 12px 12px 0;
        margin-bottom: 0.5rem;
        max-width: 80%;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
