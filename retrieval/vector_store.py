"""
Vector Store: Local vector index with strict metadata filtering.
Ensures document isolation so Customer A's data is never retrieved for Customer B.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from retrieval.embeddings import embedding_engine
from utils.logger import get_logger

logger = get_logger("vector_store")


class DocumentChunk:
    def __init__(self, text: str, metadata: Dict[str, Any], vector: List[float] = None):
        self.text = text
        self.metadata = metadata
        self.vector = vector


class LocalVectorStore:
    def __init__(self):
        self.chunks: List[DocumentChunk] = []

    def add_chunks(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """Embeds and indexes document chunks."""
        if not texts:
            return

        embeddings = embedding_engine.embed_documents(texts)
        for text, meta, emb in zip(texts, metadatas, embeddings):
            chunk = DocumentChunk(text=text, metadata=meta, vector=emb)
            self.chunks.append(chunk)

        logger.info(f"Indexed {len(texts)} chunks. Total in store: {len(self.chunks)}")

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k relevant chunks matching query and passing metadata filter.
        """
        if not self.chunks:
            return []

        # 1. Apply strict metadata filtering first
        filtered_chunks = self.chunks
        if filter_metadata:
            filtered_chunks = [
                c for c in self.chunks
                if all(c.metadata.get(mk) == mv for mk, mv in filter_metadata.items())
            ]

        if not filtered_chunks:
            return []

        # 2. Embed query
        query_vec = np.array(embedding_engine.embed_query(query), dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        # 3. Compute cosine similarities
        scored_chunks = []
        for c in filtered_chunks:
            c_vec = np.array(c.vector, dtype=np.float32)
            c_norm = np.linalg.norm(c_vec)
            if c_norm > 0:
                c_vec = c_vec / c_norm

            if query_vec.shape != c_vec.shape:
                # Dimension mismatch safeguard (e.g. if chunks or query came from different fallback states)
                q_words = set(query.lower().split())
                c_words = set(c.text.lower().split())
                intersection = len(q_words & c_words)
                union = len(q_words | c_words) or 1
                similarity = float(intersection / union)
            else:
                similarity = float(np.dot(query_vec, c_vec))
            scored_chunks.append((similarity, c))

        # Sort descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, chunk in scored_chunks[:k]:
            results.append({
                "text": chunk.text,
                "metadata": chunk.metadata,
                "score": score,
            })

        return results

    def clear_document(self, document_id: str) -> None:
        """Remove chunks associated with a specific document."""
        self.chunks = [c for c in self.chunks if c.metadata.get("document_id") != document_id]


# Global vector store
vector_store = LocalVectorStore()
