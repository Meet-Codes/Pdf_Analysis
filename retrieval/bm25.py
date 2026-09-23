"""
BM25 Keyword Retrieval: Self-contained Okapi BM25 indexer.
Provides fast, deterministic lexical search without external C-compiler dependencies.
"""

import math
import re
from typing import List, Dict, Any, Tuple


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lengths: List[int] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_metadatas: List[Dict[str, Any]] = []
        self.doc_texts: List[str] = []
        self.term_freqs: List[Dict[str, int]] = []
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-z0-9/\-]{2,}\b", text.lower())

    def index_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        """Builds BM25 index from document texts and metadatas."""
        self.doc_texts = texts
        self.doc_metadatas = metadatas
        self.corpus_size = len(texts)
        if self.corpus_size == 0:
            return

        self.doc_tokens = []
        self.doc_lengths = []
        self.term_freqs = []
        self.doc_freqs = {}

        total_tokens = 0
        for text in texts:
            tokens = self._tokenize(text)
            self.doc_tokens.append(tokens)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_tokens += doc_len

            # TF and DF
            tf: Dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            self.term_freqs.append(tf)

            for unique_tok in tf.keys():
                self.doc_freqs[unique_tok] = self.doc_freqs.get(unique_tok, 0) + 1

        self.avg_doc_len = total_tokens / float(self.corpus_size) if self.corpus_size > 0 else 0.0

        # Calculate IDFs
        self.idf = {}
        for term, df in self.doc_freqs.items():
            # Standard Lucene / Okapi IDF
            idf_score = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))
            self.idf[term] = max(0.01, idf_score)

    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Returns top-k documents matching query with metadata filtering."""
        if self.corpus_size == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: List[Tuple[float, int]] = []

        for idx in range(self.corpus_size):
            # Apply metadata filter
            if filter_metadata:
                meta = self.doc_metadatas[idx]
                if not all(meta.get(fk) == fv for fk, fv in filter_metadata.items()):
                    continue

            doc_len = self.doc_lengths[idx]
            tf_dict = self.term_freqs[idx]
            score = 0.0

            len_norm = 1.0 - self.b + self.b * (doc_len / self.avg_doc_len if self.avg_doc_len > 0 else 1.0)

            for q_tok in query_tokens:
                if q_tok in tf_dict:
                    freq = tf_dict[q_tok]
                    term_idf = self.idf.get(q_tok, 0.0)
                    term_score = term_idf * ((freq * (self.k1 + 1.0)) / (freq + self.k1 * len_norm))
                    score += term_score

            if score > 0.0:
                scores.append((score, idx))

        scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc_idx in scores[:k]:
            results.append({
                "text": self.doc_texts[doc_idx],
                "metadata": self.doc_metadatas[doc_idx],
                "score": score,
                "engine": "bm25",
            })

        return results


# Global BM25 Index
bm25_index = BM25Index()
