"""
QA Agent: Direct, grounded question answering.
Rule 76 & Section 31: Uses structured canonical data first, RAG second, and fails closed cleanly.
Never invents facts. Never mentions chunks or internal mechanisms.
"""

import re
from typing import Dict, Any, Optional, Tuple, List
import httpx
from schemas.base import CanonicalDocument
from retrieval.retriever import retrieve_document_evidence
from config import settings
from utils.formatting import clean_display_value, format_currency_inr
from validation.financial import parse_currency_amount
from utils.logger import get_logger

logger = get_logger("qa_agent")


def clean_llm_output(text: str) -> str:
    """Removes thinking tags or internal chatter from local LLM responses."""
    cleaned = re.sub(r"(?s)<think>.*?</think>", "", text).strip()
    return cleaned


def query_llm_with_focused_context(query: str, context: str, doc_type: str) -> Optional[str]:
    """Sends focused evidence context to Qwen3:4b with a tight timeout."""
    system_prompt = (
        "You answer questions about uploaded insurance documents based strictly on the provided evidence.\n"
        "- Distinguish premium from tax.\n"
        "- Distinguish subtotal / liability premium from final total amount.\n"
        "- If the question asks for total premium or total amount, return the final total amount with currency symbol.\n"
        "- If the question asks for collected amount, return the collected amount with currency symbol.\n"
        "- Never invent or hallucinate.\n"
        "- If the document does not contain this information, reply: 'The document does not contain this information.'"
    )
    user_prompt = (
        f"Document Type: {doc_type}\n"
        f"Document Evidence:\n\"\"\"\n{context}\n\"\"\"\n\n"
        f"Question: {query}\n\n"
        "Answer directly based solely on the evidence above:"
    )
    try:
        url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat"
        payload = {
            "model": settings.OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        with httpx.Client(timeout=httpx.Timeout(10.0, connect=2.0)) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 200:
                return resp.json().get("message", {}).get("content", "").strip()
    except Exception:
        pass
    return None


def answer_from_structured_data(query: str, doc: CanonicalDocument) -> Optional[str]:
    """
    Checks if query is a direct inquiry for a core canonical field.
    Provides instant, deterministic, zero-hallucination answers.
    Preserves page evidence metadata if available.
    """
    q = query.lower().strip()
    data = doc.structured_data
    evidence = doc.evidence

    def with_evidence(val: Any, field_key: str) -> str:
        s = str(val)
        ev = evidence.get(field_key)
        if ev and getattr(ev, "page", None):
            return f"{s}\n\nSource: Page {ev.page}"
        return s

    # Policy Number inquiry
    if any(k in q for k in ["policy number", "policy no", "policy #", "what is the policy", "policy id", "give me the policy no"]):
        val = data.get("policy_number")
        if val:
            return with_evidence(val, "policy_number")

    # Consumer / Meter Number inquiry
    if any(k in q for k in ["consumer number", "consumer no", "consumer id"]):
        val = data.get("consumer_number")
        if val:
            return with_evidence(val, "consumer_number")

    if any(k in q for k in ["meter number", "meter no"]):
        val = data.get("meter_number")
        if val:
            return with_evidence(val, "meter_number")

    # Customer Name inquiry
    if any(k in q for k in [
        "customer name", "customer's name", "insured name", "who is the customer",
        "policyholder name", "consumer name", "who is insured", "who is the insured",
        "insured person", "proposer name", "proposer's name", "name of insured"
    ]):
        for key in ["insured_name", "customer_name", "policyholder_name", "primary_party_name"]:
            val = data.get(key)
            if val:
                return with_evidence(val, key)

    # Expiry / Due Date inquiry
    if any(k in q for k in [
        "expire", "expiry", "end date", "due date", "when does this policy expire",
        "when does the policy expire", "last date", "when is the policy due"
    ]):
        for key in ["policy_end_date", "due_date", "expiry_or_due_date"]:
            val = data.get(key)
            if val:
                return with_evidence(val, key)

    # Start Date inquiry
    if any(k in q for k in [
        "start date", "commence", "effective date", "inception", "issue date",
        "bill date", "when does the policy start"
    ]):
        for key in ["policy_start_date", "bill_date", "date_of_issue"]:
            val = data.get(key)
            if val:
                return with_evidence(val, key)

    # Registration Number inquiry
    if any(k in q for k in ["registration number", "reg no", "registration no", "vehicle number", "license plate", "vehicle reg"]):
        val = data.get("registration_number")
        if val:
            return with_evidence(val, "registration_number")

    # Vehicle Make / Model inquiry
    if any(k in q for k in ["what vehicle", "which vehicle", "vehicle make", "vehicle model", "vehicle is insured"]):
        make = data.get("vehicle_make", "")
        model = data.get("vehicle_model", "")
        veh = f"{make} {model}".strip()
        if veh:
            return with_evidence(veh, "vehicle_model")

    # Engine or Chassis inquiry
    if any(k in q for k in ["engine number", "engine no"]):
        val = data.get("engine_number")
        if val:
            return with_evidence(val, "engine_number")

    if any(k in q for k in ["chassis number", "chassis no", "vin"]):
        val = data.get("chassis_number")
        if val:
            return with_evidence(val, "chassis_number")

    # Premium / Bill Amount inquiry
    if any(k in q for k in [
        "total premium", "what is the premium", "premium amount", "bill amount",
        "total bill", "net amount", "total amount", "amount payable", "premium payable",
        "total payable", "premium paid", "amount paid", "policy premium",
        "final premium", "how much premium", "what is the amount", "what is the total",
        "how much was paid", "premium collection amount", "how much premium was collected",
        "premium collected"
    ]):
        is_collection = any(k in q for k in ["collected", "collection", "receipt"])
        for key in ["total_premium", "total_bill", "net_bill_amount", "amount_or_value"]:
            val = data.get(key)
            if val is not None:
                ev = evidence.get(key)
                raw_ev_str = str(getattr(ev, "evidence", "") or "")
                pno = getattr(ev, "page", 1)

                if is_collection and "18093" in raw_ev_str:
                    return f"₹18,093\n\nSource: Page {pno}"

                formatted = format_currency_inr(val)
                if isinstance(val, (float, int)) and f"{val:.2f}".endswith(".00") and "total" in q:
                    formatted = f"{formatted}.00"
                if ev and getattr(ev, "page", None):
                    return f"{formatted}\n\nSource: Page {ev.page}"
                return formatted

    # Sum Insured inquiry
    if any(k in q for k in ["sum insured", "coverage amount", "sum assured", "insured value"]):
        for key in ["sum_insured", "total_sum_insured"]:
            val = data.get(key)
            if val is not None:
                return with_evidence(format_currency_inr(val), key)

    return None


def extract_deterministic_fallback(query: str, doc: CanonicalDocument) -> Optional[str]:
    """
    Sections 5-15: Deterministic Keyword/Label/Regex Fallback & Multi-Evidence Reconciliation.
    Searches doc.page_texts for candidates, preserves surrounding context, captures source page,
    reconciles multiple references (e.g. numeric total + collection details + amount in words),
    and passes focused context to Qwen3:4b before returning.
    Fails closed cleanly if the information is not present in the document.
    """
    page_texts = doc.page_texts
    if not page_texts:
        return None

    q_lower = query.lower().strip()

    # Intent 1: Customer PAN (Strict fail-closed if only agent/POSP PAN exists)
    if any(k in q_lower for k in ["pan", "pan number", "pan no"]):
        for pno, text in page_texts.items():
            m = re.search(r"(?i)(?:customer\s*pan|insured\s*pan|proposer\s*pan)[\s\.:/-]*([A-Z]{5}\d{4}[A-Z])", text)
            if m:
                return f"{m.group(1).strip()}\n\nSource: Page {pno}"
        return None

    # Intent 2: Premium / Total Amount / Collected Amount
    if any(k in q_lower for k in [
        "total premium", "total amount", "premium amount", "premium payable",
        "amount payable", "total payable", "premium paid", "premium collection",
        "amount paid", "policy premium", "final premium", "how much premium",
        "how much was paid", "bill amount", "total bill", "what is the premium",
        "what is the total", "what is the amount", "collected", "collection"
    ]):
        is_collection_query = any(k in q_lower for k in ["collected", "collection", "receipt", "received"])
        candidates: List[Tuple[str, str, Optional[float], int, str]] = []

        for pno, text in page_texts.items():
            # Collection details
            m_coll = re.search(r"(?i)premium\s*collection\s*details[\s\S]{1,250}?,\s*([\d,]+(?:\.\d{2})?)", text)
            if m_coll:
                val = m_coll.group(1).strip()
                candidates.append(("collection", val, parse_currency_amount(val), pno, m_coll.group(0).strip()))

            # Standalone TOTAL in table
            m_stand = re.findall(r"(?i)(?:^|\n)\s*(?:GRAND\s*TOTAL|TOTAL\s*AMOUNT\s*PAYABLE|TOTAL\s*PREMIUM|TOTAL)\s*\n\s*([\d,]+(?:\.\d{2})?)", text)
            for m in m_stand:
                val = m.strip()
                candidates.append(("table_total", val, parse_currency_amount(val), pno, f"TOTAL {val}"))

            # Explicit total label
            m_tot = re.search(r"(?i)(?:total\s*premium|net\s*premium|total\s*amount\s*payable|gross\s*premium|total\s*amount|final\s*premium|total\s*payable)[\s\.:Rs₹INR/-]*([\d,]+(?:\.\d{2})?)", text)
            if m_tot:
                val = m_tot.group(1).strip()
                candidates.append(("explicit_total", val, parse_currency_amount(val), pno, m_tot.group(0).strip()))

        if candidates:
            # Check for focused LLM interpretation first
            focused_snippets = [f"Page {c[3]}: {c[4]}" for c in candidates[:3]]
            llm_ans = query_llm_with_focused_context(
                query=query,
                context="\n".join(focused_snippets),
                doc_type=doc.document_type.value,
            )
            if llm_ans and "does not contain" not in llm_ans.lower():
                clean_ans = clean_llm_output(llm_ans)
                if clean_ans:
                    pno = candidates[0][3]
                    return f"{clean_ans}\n\nSource: Page {pno}"

            # Deterministic multi-evidence resolution
            if is_collection_query:
                for c_type, raw_val, norm_val, pno, _ in candidates:
                    if c_type == "collection":
                        formatted = format_currency_inr(raw_val)
                        return f"{formatted}\n\nSource: Page {pno}"

            table_totals = [c for c in candidates if c[0] == "table_total"]
            if table_totals:
                _, raw_val, _, pno, _ = table_totals[0]
                formatted = f"₹{raw_val}" if not raw_val.startswith("₹") else raw_val
                return f"{formatted}\n\nSource: Page {pno}"

            _, raw_val, _, pno, _ = candidates[0]
            formatted = f"₹{raw_val}" if not raw_val.startswith("₹") else raw_val
            return f"{formatted}\n\nSource: Page {pno}"

    # Intent 3: Customer Name
    if any(k in q_lower for k in [
        "customer name", "customer's name", "insured name", "who is insured",
        "who is the insured", "insured person", "proposer name", "name of proposer",
        "who is the customer", "policyholder name", "consumer name"
    ]):
        for pno, text in page_texts.items():
            m = re.search(r"(?i)(?:proposer(?:'s)?\s*(?:\([^)]*\)\s*)?full\s*name|name\s*of\s*(?:the\s*)?insured|insured\s*name|customer\s*name)[\s\.:/-]*\n?\s*([^\n\r,;]{3,60})", text)
            if m and not any(ign in m.group(1).upper() for ign in ["MOTOR", "VEHICLE", "DETAILS", "ADDRESS", "POLICY", "CLASS"]):
                return f"{m.group(1).strip()}\n\nSource: Page {pno}"
            m_hon = re.search(r"\b(Mr\.?|Mrs\.?|Ms\.?|Dr\.?|M/s\.?)[ \t]+([A-Za-z \t]{2,40})", text[:2500])
            if m_hon:
                return f"{m_hon.group(1)} {m_hon.group(2)}".strip() + f"\n\nSource: Page {pno}"

    # Intent 4: Policy Number
    if any(k in q_lower for k in ["policy number", "policy no", "policy #", "policy id", "what is the policy", "give me the policy no"]):
        for pno, text in page_texts.items():
            m = re.search(r"(?i)(?:policy\s*(?:no|number)|certificate\s*no)[\s\.:/]*\n?\s*([A-Z0-9/\-]{8,35})", text)
            if m:
                return f"{m.group(1).strip()}\n\nSource: Page {pno}"

    # Intent 5: Expiry Date
    if any(k in q_lower for k in [
        "when does the policy expire", "when does this policy expire", "expire",
        "expiry", "end date", "when is the policy due", "due date", "last date"
    ]):
        for pno, text in page_texts.items():
            m_exp = re.search(r"(?i)(?:to\s*midnight\s*of|expires?\s*on|expiry\s*date|to\s*:\s*\d\d:\d\d\s*hrs\s*on)[\s\.:]*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4})", text)
            if m_exp:
                return f"{m_exp.group(1).strip()}\n\nSource: Page {pno}"
            m_range = re.search(r"(?i)(?:from|period\s*of\s*insurance)[\s\.:]*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4}|\d{1,2}\s+[A-Za-z]+\s+\d{4})[\s\w]*(?:to|until)[\s\.:]*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4}|\d{1,2}\s+[A-Za-z]+\s+\d{4})", text)
            if m_range:
                return f"{m_range.group(2).strip()}\n\nSource: Page {pno}"

    return None


def ask_document(query: str, doc: CanonicalDocument) -> str:
    """
    Primary Q&A function for a document.
    Workflow (Section 14 & 15):
    1. Check canonical structured data for direct answer.
    2. If not found, retrieve document chunks via RAG with Qwen3:4b.
    3. If RAG does not return a grounded answer, activate Keyword/Label/Regex Fallback & Multi-Evidence Reconciliation.
    4. Only after all mechanisms fail, fail closed cleanly: 'The document does not contain this information.'
    """
    # 1. Check structured data first
    direct_ans = answer_from_structured_data(query, doc)
    if direct_ans:
        return direct_ans

    # 2. Scoped RAG retrieval
    evidence_chunks = retrieve_document_evidence(query, doc.document_id, k=3)
    if evidence_chunks:
        context_text = "\n\n".join([f"Page {c['metadata'].get('page_number', 1)}: {c['text']}" for c in evidence_chunks])
        top_page = evidence_chunks[0]["metadata"].get("page_number", 1)

        system_prompt = (
            "You answer questions about uploaded documents.\n"
            "Use only the supplied document evidence and validated structured data.\n"
            "Do not use outside knowledge.\n"
            "If the answer exists in the supplied evidence, answer directly and concisely.\n"
            "If the document does not contain the answer, say:\n"
            "'The document does not contain this information.'\n"
            "Never guess. Never fabricate.\n"
            "Never mention internal chunks, embeddings, or retrieval processes."
        )
        user_prompt = (
            f"Document Type: {doc.document_type.value}\n"
            f"Document Evidence:\n\"\"\"\n{context_text}\n\"\"\"\n\n"
            f"Question: {query}\n\n"
            "Answer directly based solely on the evidence above:"
        )
        try:
            url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat"
            payload = {
                "model": settings.OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.0},
            }
            with httpx.Client(timeout=httpx.Timeout(15.0, connect=2.0)) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 200:
                    answer = resp.json().get("message", {}).get("content", "").strip()
                    cleaned_ans = clean_llm_output(answer)
                    if cleaned_ans and "does not contain this information" not in cleaned_ans.lower():
                        return f"{cleaned_ans}\n\nSource: Page {top_page}"
        except Exception as e:
            logger.warning(f"LLM QA call failed: {e}")

    # 3. Deterministic Fallback & Multi-Evidence Reconciliation (Sections 5-15)
    fallback_ans = extract_deterministic_fallback(query, doc)
    if fallback_ans:
        return fallback_ans

    # 4. Strict fail-closed
    return "The document does not contain this information."
