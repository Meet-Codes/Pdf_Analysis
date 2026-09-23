"""
Vector Store: Local vector index with strict metadata filtering.
Ensures document isolation so Customer A's data is never retrieved for Customer B.
Uses genuine semantic cosine similarity.
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
        """Embeds and indexes document chunks with genuine semantic embeddings."""
        if not texts:
            return

        embeddings = embedding_engine.embed_documents(texts)
        provider_info = embedding_engine.get_provider_info()

        for text, meta, emb in zip(texts, metadatas, embeddings):
            chunk_meta = dict(meta)
            chunk_meta["embedding_provider"] = provider_info["provider"]
            chunk_meta["embedding_dim"] = len(emb)
            chunk = DocumentChunk(text=text, metadata=chunk_meta, vector=emb)
            self.chunks.append(chunk)

        logger.info(f"Indexed {len(texts)} chunks into VectorStore ({provider_info['provider']}, dim={len(embeddings[0]) if embeddings else 0}). Total: {len(self.chunks)}")

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k relevant chunks matching query and passing metadata filter.
        Guarantees strict semantic vector scoring.
        """
        if not self.chunks:
            return []

        # 1. Apply strict metadata filtering first (Document Isolation)
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

        # Check for dimension alignment
        target_dim = len(query_vec)
        first_chunk_dim = len(filtered_chunks[0].vector) if filtered_chunks[0].vector else 0

        if first_chunk_dim != target_dim and first_chunk_dim > 0:
            # Dimension mismatch (e.g. index was created with different provider than current query)
            logger.warning(
                f"Dimension alignment needed: query dim={target_dim}, indexed dim={first_chunk_dim}. "
                "Re-embedding document chunks with current semantic provider for precision."
            )
            chunk_texts = [c.text for c in filtered_chunks]
            new_embs = embedding_engine.embed_documents(chunk_texts)
            for c, new_emb in zip(filtered_chunks, new_embs):
                c.vector = new_emb

        # 3. Compute cosine similarities
        scored_chunks = []
        for c in filtered_chunks:
            c_vec = np.array(c.vector, dtype=np.float32)
            c_norm = np.linalg.norm(c_vec)
            if c_norm > 0:
                c_vec = c_vec / c_norm

            if query_vec.shape == c_vec.shape:
                similarity = float(np.dot(query_vec, c_vec))
            else:
                similarity = 0.0

            scored_chunks.append((similarity, c))

        # Sort descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, chunk in scored_chunks[:k]:
            results.append({
                "text": chunk.text,
                "metadata": chunk.metadata,
                "score": float(round(score, 4)),
                "engine": "vector",
            })

        return results

    def clear_document(self, document_id: str) -> None:
        """Remove chunks associated with a specific document."""
        self.chunks = [c for c in self.chunks if c.metadata.get("document_id") != document_id]


# Global vector store
vector_store = LocalVectorStore()
