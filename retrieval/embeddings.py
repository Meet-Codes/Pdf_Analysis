"""
Embeddings Provider: Genuine Semantic Embedding Engine.
Supports preferred Ollama embedding endpoint (e.g. qwen3-embedding:0.6b)
with automatic fallback to local SentenceTransformer (all-MiniLM-L6-v2).
Zero fake/hash vector fallbacks.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import httpx
from config import settings
from utils.logger import get_logger

logger = get_logger("embeddings")


class EmbeddingEngine:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.fallback_model_name = getattr(settings, "LOCAL_EMBEDDING_FALLBACK", "all-MiniLM-L6-v2")
        self._sentence_transformer = None
        self.active_provider: str = "unknown"
        self.dim: int = 1024  # Standard default for qwen3-embedding:0.6b

    def _get_local_model(self):
        """Lazy-load the local SentenceTransformer fallback model."""
        if self._sentence_transformer is None:
            logger.info(f"Loading local semantic embedding fallback model '{self.fallback_model_name}'...")
            from sentence_transformers import SentenceTransformer
            self._sentence_transformer = SentenceTransformer(self.fallback_model_name)
            logger.info(f"Local semantic embedding model '{self.fallback_model_name}' ready.")
        return self._sentence_transformer

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute semantic embeddings for a batch of texts."""
        if not texts:
            return []

        # Attempt 1: Query Ollama embedding endpoint
        try:
            url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/embed"
            payload = {
                "model": self.model_name,
                "input": texts,
            }
            timeout_val = min(15.0, float(getattr(settings, "OLLAMA_TIMEOUT", 15)))
            with httpx.Client(timeout=httpx.Timeout(timeout_val, connect=2.0)) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    embeddings = data.get("embeddings")
                    if embeddings and len(embeddings) == len(texts):
                        self.active_provider = f"ollama ({self.model_name})"
                        if embeddings[0]:
                            self.dim = len(embeddings[0])
                        logger.info(f"Generated {len(texts)} semantic embeddings via Ollama [{self.model_name}] (dim={self.dim})")
                        # Normalize vectors
                        normalized = []
                        for vec in embeddings:
                            arr = np.array(vec, dtype=np.float32)
                            norm = np.linalg.norm(arr)
                            if norm > 0:
                                arr = arr / norm
                            normalized.append(arr.tolist())
                        return normalized
                else:
                    logger.warning(f"Ollama embed returned status {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"Ollama semantic embedding unavailable ({e}). Triggering local SentenceTransformer fallback.")

        # Attempt 2: Local SentenceTransformer Fallback (Real semantic embeddings, never hash)
        try:
            local_model = self._get_local_model()
            embeddings_arr = local_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            self.active_provider = f"local ({self.fallback_model_name})"
            if len(embeddings_arr) > 0:
                self.dim = int(embeddings_arr[0].shape[0])
            logger.info(f"Generated {len(texts)} semantic embeddings via local SentenceTransformer [{self.fallback_model_name}] (dim={self.dim})")
            return [vec.tolist() for vec in embeddings_arr]
        except Exception as e:
            logger.error(f"Critical error: Both Ollama and local SentenceTransformer embedding failed: {e}", exc_info=True)
            raise RuntimeError(f"Semantic embedding generation failed for both Ollama and local model: {e}")

    def embed_query(self, text: str) -> List[float]:
        """Compute semantic embedding for a single search query."""
        results = self.embed_documents([text])
        return results[0] if results else [0.0] * self.dim

    def get_provider_info(self) -> Dict[str, Any]:
        """Returns details about active embedding provider and dimension."""
        return {
            "provider": self.active_provider,
            "ollama_model": self.model_name,
            "fallback_model": self.fallback_model_name,
            "dimension": self.dim,
        }


# Global singleton
embedding_engine = EmbeddingEngine()
