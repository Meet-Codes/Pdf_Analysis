"""
Retriever: Indexes canonical documents and provides scoped question-answering retrieval.
"""

from typing import List, Dict, Any, Optional
from schemas.base import CanonicalDocument
from retrieval.vector_store import vector_store
from utils.logger import get_logger

logger = get_logger("retriever")


def index_canonical_document(
    canonical: CanonicalDocument,
    page_texts: Dict[int, str],
) -> None:
    """
    Chunks and indexes a document with rich metadata into the vector store.
    """
    texts = []
    metadatas = []

    doc_id = canonical.document_id
    fname = canonical.file_name
    dtype = canonical.document_type.value
    customer = str(canonical.structured_data.get("insured_name") or canonical.structured_data.get("customer_name") or "")
    policy_no = str(canonical.structured_data.get("policy_number") or canonical.structured_data.get("consumer_number") or "")

    for pno, text in page_texts.items():
        if not text or not text.strip():
            continue

        # Split into logical paragraphs / chunks (~300-500 chars)
        raw_paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks_for_page = []
        for p in raw_paras:
            if len(p) > 500:
                # Subdivide by lines with sliding window
                lines = [line.strip() for line in p.split("\n") if line.strip()]
                curr = []
                curr_len = 0
                for line in lines:
                    curr.append(line)
                    curr_len += len(line) + 1
                    if curr_len >= 350:
                        chunks_for_page.append("\n".join(curr))
                        curr = curr[-2:] if len(curr) > 2 else []  # slight overlap
                        curr_len = sum(len(x) + 1 for x in curr)
                if curr:
                    chunks_for_page.append("\n".join(curr))
            else:
                chunks_for_page.append(p)

        for idx, para in enumerate(chunks_for_page):
            texts.append(para)
            metadatas.append({
                "document_id": doc_id,
                "file_name": fname,
                "page_number": pno,
                "document_type": dtype,
                "customer": customer,
                "policy_number": policy_no,
                "chunk_index": idx,
            })

    if texts:
        vector_store.add_chunks(texts, metadatas)
        logger.info(f"Indexed document {doc_id} with {len(texts)} chunks.")


def retrieve_document_evidence(
    query: str,
    document_id: str,
    k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Retrieves evidence strictly scoped to the specified document_id.
    Prevents cross-document data leakage.
    """
    return vector_store.similarity_search(
        query=query,
        k=k,
        filter_metadata={"document_id": document_id},
    )
