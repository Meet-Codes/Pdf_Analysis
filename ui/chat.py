"""
Chat UI: Conversational grounded Q&A interface for active documents.
Adheres strictly to Rule 76: Only answers from document, fails closed cleanly.
"""

from typing import List, Dict
import streamlit as st
from schemas.base import CanonicalDocument
from agents.qa_agent import ask_document


def render_document_chat(active_doc: CanonicalDocument):
    """Renders the document Q&A panel with suggested questions and conversational history."""
    st.markdown(
        """
        <div class="card-header">
            <span>💬</span> <span>Ask Document</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    doc_id = active_doc.document_id
    session_key = f"chat_history_{doc_id}"

    if session_key not in st.session_state:
        st.session_state[session_key] = []

    # Quick prompt suggestion buttons
    st.caption("Suggested Questions:")
    quick_cols = st.columns(3)
    suggested_q = None

    with quick_cols[0]:
        if st.button("What is the policy / consumer number?", key=f"q1_{doc_id}", use_container_width=True):
            suggested_q = "What is the policy or consumer number?"
    with quick_cols[1]:
        if st.button("When does this expire / due date?", key=f"q2_{doc_id}", use_container_width=True):
            suggested_q = "When does this policy expire or when is the due date?"
    with quick_cols[2]:
        if st.button("What is the total amount / premium?", key=f"q3_{doc_id}", use_container_width=True):
            suggested_q = "What is the total amount or total premium?"

    # Chat history rendering
    for msg in st.session_state[session_key]:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            st.markdown(f'<div class="chat-user">{content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-assistant">{content}</div>', unsafe_allow_html=True)

    # Input form
    with st.form(key=f"chat_form_{doc_id}", clear_on_submit=True):
        user_input = st.text_input(
            "Ask a question about this document:",
            placeholder="e.g. What is the customer's name?",
            value=suggested_q if suggested_q else "",
        )
        submit_clicked = st.form_submit_button("Ask")

    query_to_process = suggested_q if suggested_q else (user_input if submit_clicked else None)

    if query_to_process and query_to_process.strip():
        # Record user query
        st.session_state[session_key].append({"role": "user", "content": query_to_process})

        # Answer strictly from document
        with st.spinner("Analyzing document evidence..."):
            answer = ask_document(query_to_process, active_doc)

        st.session_state[session_key].append({"role": "assistant", "content": answer})
        st.rerun()
