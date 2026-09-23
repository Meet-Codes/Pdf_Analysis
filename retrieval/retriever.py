"""
Retriever: Indexes canonical documents with structural chunking and table-aware representations.
Provides scoped question-answering retrieval without cross-document leakage.
"""

from typing import List, Dict, Any, Optional
import re
from schemas.base import CanonicalDocument
from retrieval.hybrid_retriever import hybrid_retriever
from utils.logger import get_logger

logger = get_logger("retriever")


def format_table_to_semantic_chunk(table: Dict[str, Any], page_no: int) -> str:
    """
    Converts extracted table structures into first-class semantic retrieval representation.
    Preserves headers, rows, key-values, and relational semantics without arbitrary splitting.
    Example:
      TABLE: PREMIUM COMPUTATION (Page 1, Table 0)
      Headers: Component | Rate | Amount
      Row 1: Basic Premium = ₹15,430
      Row 2: GST (18%) = ₹2,777
      Row 3: Total Premium = ₹18,207
    """
    lines = []
    t_idx = table.get("table_index", 0)
    headers = [str(h).strip() for h in table.get("headers", []) if h]
    rows = table.get("rows", [])
    kv_map = table.get("key_value_map", {})

    # Detect title if available or default
    table_title = table.get("title") or (headers[0] if headers and len(headers) == 1 else "TABLE")
    lines.append(f"TABLE: {table_title.upper()} (Page {page_no}, Table {t_idx})")

    if headers and len(headers) > 1:
        lines.append(f"Headers: {' | '.join(headers)}")

    if kv_map:
        for k, v in kv_map.items():
            if str(k).strip() and str(v).strip():
                lines.append(f"{str(k).strip()} = {str(v).strip()}")
    elif rows:
        for r_idx, row in enumerate(rows):
            row_items = []
            for c_idx, cell in enumerate(row):
                cell_val = str(cell).strip()
                if not cell_val:
                    continue
                header_name = headers[c_idx] if c_idx < len(headers) and headers[c_idx] else f"Col_{c_idx+1}"
                row_items.append(f"{header_name}: {cell_val}")
            if row_items:
                if len(row) >= 2 and row[0]:
                    key_val = str(row[0]).strip()
                    other_vals = [str(c).strip() for c in row[1:] if str(c).strip()]
                    lines.append(f"Row {r_idx + 1}: {key_val} = {' | '.join(other_vals)}")
                else:
                    lines.append(f"Row {r_idx + 1}: {', '.join(row_items)}")

    return "\n".join(lines)


def structural_chunk_page(
    text: str,
    target_token_count: int = 600,
    overlap_tokens: int = 80,
) -> List[Dict[str, Any]]:
    """
    Structure-aware semantic chunking:
    - Retains headings with their content
    - Keeps key-value pairs contiguous
    - Approximates 500-700 tokens per chunk with 75-100 tokens overlap
    - Prevents arbitrary character-only breaks across critical fields
    """
    if not text or not text.strip():
        return []

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return []

    # Rough approximation: 1 token ≈ 4 characters
    target_char_count = target_token_count * 4
    overlap_char_count = overlap_tokens * 4

    chunks: List[Dict[str, Any]] = []
    current_lines: List[str] = []
    current_chars = 0
    current_heading = ""

    def is_heading(line: str) -> bool:
        if len(line) < 3 or len(line) > 80:
            return False
        # Uppercase header or ending with colon
        if line.isupper() and not any(char.isdigit() for char in line[:5]):
            return True
        if line.endswith(":") and not any(delm in line for delm in ["rs.", "inr", "₹"]):
            return True
        if re.match(r"(?i)^(?:section|schedule|policy|certificate|details|coverage|charges|bill)\b", line):
            return True
        return False

    for line in lines:
        if is_heading(line):
            current_heading = line

        current_lines.append(line)
        current_chars += len(line) + 1

        if current_chars >= target_char_count:
            chunk_content = "\n".join(current_lines)
            chunks.append({
                "text": chunk_content,
                "section_heading": current_heading,
                "chunk_type": "text",
            })

            # Overlap: keep last few lines proportional to overlap_char_count
            overlap_lines = []
            overlap_chars = 0
            for prev_line in reversed(current_lines):
                overlap_lines.insert(0, prev_line)
                overlap_chars += len(prev_line) + 1
                if overlap_chars >= overlap_char_count:
                    break

            # If the active heading is not already in the overlap, keep heading context
            if current_heading and current_heading not in overlap_lines:
                overlap_lines.insert(0, f"[{current_heading}]")

            current_lines = list(overlap_lines)
            current_chars = sum(len(l) + 1 for l in current_lines)

    if current_lines:
        chunk_content = "\n".join(current_lines)
        # Avoid tiny tail chunks by appending to previous if very small (< 100 chars)
        if len(chunk_content) < 120 and chunks:
            chunks[-1]["text"] = f"{chunks[-1]['text']}\n{chunk_content}"
        else:
            chunks.append({
                "text": chunk_content,
                "section_heading": current_heading,
                "chunk_type": "text",
            })

    return chunks


def index_canonical_document(
    canonical: CanonicalDocument,
    page_texts: Dict[int, str],
    tables: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Structural indexing of canonical documents with first-class table representation
    into the hybrid store (BM25 + Vector Store).
    """
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    doc_id = canonical.document_id
    fname = canonical.file_name
    dtype = canonical.document_type.value
    customer = str(canonical.structured_data.get("insured_name") or canonical.structured_data.get("customer_name") or "")
    policy_no = str(canonical.structured_data.get("policy_number") or canonical.structured_data.get("consumer_number") or "")

    # 1. Index First-Class Tables Separately (Critical Rule 2)
    doc_tables = tables if tables is not None else getattr(canonical, "tables", [])
    for t_idx, tbl in enumerate(doc_tables):
        pno = tbl.get("page_number", 1)
        semantic_table_text = format_table_to_semantic_chunk(tbl, page_no=pno)
        if semantic_table_text.strip():
            texts.append(semantic_table_text)
            metadatas.append({
                "document_id": doc_id,
                "file_name": fname,
                "page_number": pno,
                "document_type": dtype,
                "customer": customer,
                "policy_number": policy_no,
                "chunk_index": len(texts) - 1,
                "is_table": True,
                "table_index": t_idx,
                "source_method": "table_extractor",
            })

    # 2. Index Text Chunks using Structural Chunker (500-700 tokens, 75-100 overlap)
    for pno, text in page_texts.items():
        if not text or not text.strip():
            continue

        page_chunks = structural_chunk_page(text, target_token_count=600, overlap_tokens=80)
        for p_chunk in page_chunks:
            chunk_txt = p_chunk["text"].strip()
            if not chunk_txt:
                continue

            texts.append(chunk_txt)
            metadatas.append({
                "document_id": doc_id,
                "file_name": fname,
                "page_number": pno,
                "document_type": dtype,
                "customer": customer,
                "policy_number": policy_no,
                "chunk_index": len(texts) - 1,
                "is_table": False,
                "section_heading": p_chunk.get("section_heading", ""),
                "source_method": "structural_chunker",
            })

    if texts:
        hybrid_retriever.index_document(doc_id, texts, metadatas)
        table_count = sum(1 for m in metadatas if m.get("is_table"))
        logger.info(
            f"Indexed document {doc_id} into Hybrid Retriever: "
            f"{len(texts)} total chunks ({table_count} table chunks, {len(texts) - table_count} structural text chunks)."
        )


def retrieve_document_evidence(
    query: str,
    document_id: str,
    k: int = 4,
    search_variants: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves evidence strictly scoped to the specified document_id using Hybrid (BM25 + Vector + Rerank).
    Prevents cross-document data leakage.
    """
    return hybrid_retriever.retrieve(
        query=query,
        document_id=document_id,
        k=k,
        search_variants=search_variants,
    )
