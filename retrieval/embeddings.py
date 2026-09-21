"""
Embeddings Provider: Configurable embedding interface.
Supports Ollama embedding endpoint (e.g. qwen3-embedding:0.6b) and local fallback.
"""

from typing import List
import numpy as np
import httpx
from config import settings
from utils.logger import get_logger

logger = get_logger("embeddings")


class EmbeddingEngine:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._sentence_transformer = None
        self.dim = 1024  # Standard dimension for qwen3-embedding:0.6b

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a batch of texts."""
        if not texts:
            return []

        # Attempt 1: Query Ollama embedding endpoint
        try:
            url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/embed"
            payload = {
                "model": self.model_name,
                "input": texts,
            }
            timeout_val = float(getattr(settings, "OLLAMA_TIMEOUT", 30))
            with httpx.Client(timeout=httpx.Timeout(timeout_val, connect=2.0)) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    embeddings = data.get("embeddings")
                    if embeddings and len(embeddings) == len(texts):
                        if embeddings[0]:
                            self.dim = len(embeddings[0])
                        return embeddings
        except Exception as e:
            logger.debug(f"Ollama embedding request failed ({e}); checking fallback.")

        # Attempt 2: Fallback to lightweight local hash-based dense vectorizer
        return self._local_fallback_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """Compute embedding for a single search query."""
        results = self.embed_documents([text])
        return results[0] if results else [0.0] * self.dim

    def _local_fallback_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Fast, deterministic local hashing vectorizer matching target dimension
        ensuring RAG never fails even when Ollama is offline or switching models.
        """
        dim = self.dim
        vectors = []
        for text in texts:
            vec = np.zeros(dim, dtype=np.float32)
            words = text.lower().split()
            for w in words:
                h = hash(w) % dim
                vec[h] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec.tolist())
        return vectors


# Global singleton
embedding_engine = EmbeddingEngine()
