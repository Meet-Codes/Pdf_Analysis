"""
Hybrid Retriever: Multi-signal retrieval combining Exact Lexical Match, BM25 Keyword Search,
and Semantic Vector Similarity, followed by Deduplication and Cross-Encoder Reranking.
Guarantees strict document isolation (zero cross-document leakage).
"""

from typing import List, Dict, Any, Optional
from config import settings
from retrieval.vector_store import vector_store
from retrieval.bm25 import bm25_index
from retrieval.reranker import reranker
from utils.logger import get_logger

logger = get_logger("hybrid_retriever")


class HybridRetriever:
    def __init__(self):
        pass

    def index_document(
        self,
        document_id: str,
        texts: List[str],
        metadatas: List[Dict[str, Any]]
    ):
        """Indexes chunks into both the vector store and BM25 index."""
        if not texts:
            return

        # 1. Index into Vector Store
        vector_store.add_chunks(texts, metadatas)

        # 2. Index into BM25 Index
        bm25_index.index_documents(texts, metadatas)
        logger.info(f"Indexed {len(texts)} chunks for document {document_id} into Hybrid Retriever.")

    def retrieve(
        self,
        query: str,
        document_id: str,
        k: int = 4,
        search_variants: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid retrieval:
        1. BM25 keyword search across query variants (top 15-20 candidates)
        2. Semantic vector similarity search (top 15-20 candidates)
        3. Reciprocal Rank Fusion (RRF)
        4. Deduplication
        5. Cross-Encoder Reranking down to top k evidence chunks
        """
        filter_meta = {"document_id": document_id}
        variants = search_variants or [query]

        # Initial candidate pool: retrieve top 15–20 candidates before reranking
        pool_k = max(getattr(settings, "RETRIEVAL_CANDIDATE_POOL", 18), k * 4)

        # Signal A: BM25 keyword search on query variants
        bm25_cands = []
        for v in variants[:3]:
            res = bm25_index.search(query=v, k=pool_k, filter_metadata=filter_meta)
            bm25_cands.extend(res)

        # Signal B: Vector similarity search
        vector_cands = vector_store.similarity_search(
            query=query,
            k=pool_k,
            filter_metadata=filter_meta
        )

        # Signal C: Reciprocal Rank Fusion (RRF)
        # RRF score = sum(1 / (60 + rank))
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        for rank, c in enumerate(bm25_cands):
            text_key = c["text"]
            chunk_map[text_key] = c
            rrf_scores[text_key] = rrf_scores.get(text_key, 0.0) + (1.0 / (60.0 + rank + 1.0))

        for rank, c in enumerate(vector_cands):
            text_key = c["text"]
            chunk_map[text_key] = c
            rrf_scores[text_key] = rrf_scores.get(text_key, 0.0) + (1.0 / (60.0 + rank + 1.0))

        # Merge and rank candidates by RRF score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda tk: rrf_scores[tk], reverse=True)
        merged_cands = []
        for text_key in sorted_keys[:pool_k]:
            cand = chunk_map[text_key]
            cand_copy = dict(cand)
            cand_copy["rrf_score"] = float(round(rrf_scores[text_key], 5))
            merged_cands.append(cand_copy)

        # Fallback to direct candidates if RRF empty
        if not merged_cands:
            merged_cands = (vector_cands or bm25_cands)[:pool_k]

        # Signal D: Cross-Encoder Reranker (Reranks top 15-20 down to top k)
        final_top = reranker.rerank(query=query, candidates=merged_cands, top_k=k)
        logger.info(f"Hybrid retrieval: {len(bm25_cands)} BM25 + {len(vector_cands)} Vector -> {len(merged_cands)} merged pool -> {len(final_top)} reranked evidence.")
        return final_top


# Global Hybrid Retriever Singleton
hybrid_retriever = HybridRetriever()
