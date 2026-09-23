"""
Reranker: Genuine Semantic Cross-Encoder Reranker with Exact Identifier Boosting.
Uses ms-marco-MiniLM-L-6-v2 cross-encoder to rescore retrieved candidates.
Keeps heuristic lexical/entity signals as supplementary verification bonuses.
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np
from config import settings
from utils.logger import get_logger

logger = get_logger("reranker")


class Reranker:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or getattr(settings, "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self._cross_encoder = None
        self._load_failed = False

    def _get_cross_encoder(self):
        """Lazy-loads the cross-encoder model on first invocation."""
        if self._cross_encoder is None and not self._load_failed:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading semantic cross-encoder model '{self.model_name}'...")
                self._cross_encoder = CrossEncoder(self.model_name)
                logger.info(f"Semantic cross-encoder '{self.model_name}' successfully initialized.")
            except Exception as e:
                logger.warning(f"Could not load CrossEncoder '{self.model_name}': {e}. Using augmented heuristic reranking.")
                self._load_failed = True
        return self._cross_encoder

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Reranks merged retrieval candidates for relevance to the query.
        Returns top_k deduplicated evidence chunks.
        """
        if not candidates:
            return []

        # 1. Deduplicate identical or near-identical texts
        seen_keys = set()
        deduped_candidates: List[Dict[str, Any]] = []
        for cand in candidates:
            text = cand.get("text", "").strip()
            if not text:
                continue
            # First 25 words lowercase key
            norm_key = " ".join(text.split()[:25]).lower()
            if norm_key in seen_keys:
                continue
            seen_keys.add(norm_key)
            deduped_candidates.append(cand)

        if not deduped_candidates:
            return []

        # 2. Semantic Cross-Encoder Scoring
        ce_scores: Optional[List[float]] = None
        encoder = self._get_cross_encoder()
        if encoder is not None:
            try:
                pairs = [(query, cand.get("text", "")) for cand in deduped_candidates]
                raw_scores = encoder.predict(pairs)
                # Sigmoid normalization: 1 / (1 + exp(-score))
                ce_scores = [float(1.0 / (1.0 + np.exp(-float(s)))) for s in raw_scores]
            except Exception as e:
                logger.warning(f"CrossEncoder prediction failed ({e}); falling back to heuristic scoring.")
                ce_scores = None

        # 3. Heuristic and entity signals
        q_lower = query.lower()
        q_terms = [t for t in re.findall(r"\b[a-z0-9/\-]{2,}\b", q_lower) if t not in ("what", "is", "the", "of", "and", "in", "to", "for")]

        scored_candidates: List[Tuple[float, Dict[str, Any]]] = []

        for idx, cand in enumerate(deduped_candidates):
            text = cand.get("text", "")
            text_lower = text.lower()
            cand_copy = dict(cand)

            # Signal A: Semantic Cross-Encoder Score (Dominant signal: 75% weight)
            if ce_scores is not None and idx < len(ce_scores):
                semantic_score = ce_scores[idx]
            else:
                semantic_score = cand.get("rrf_score", 0.5)

            # Signal B: Exact substring match
            exact_bonus = 1.0 if q_lower in text_lower else 0.0

            # Signal C: Term overlap density
            matched_terms = [t for t in q_terms if t in text_lower]
            term_density = len(matched_terms) / float(len(q_terms)) if q_terms else 0.0

            # Signal D: Entity / Numeric match
            entity_bonus = 0.0
            if any(w in q_lower for w in ["amount", "premium", "total", "bill", "rs", "inr"]):
                if re.search(r"(?:rs\.?|inr|₹)?\s*[\d,]+(?:\.\d{2})?", text_lower):
                    entity_bonus += 0.3
            if any(w in q_lower for w in ["date", "expire", "start", "valid", "period"]):
                if re.search(r"\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}", text_lower) or re.search(r"\d{1,2}\s+[a-z]+\s+\d{4}", text_lower):
                    entity_bonus += 0.3
            if any(w in q_lower for w in ["number", "no", "id", "#", "vin"]):
                if re.search(r"[a-z0-9/\-]{6,30}", text_lower):
                    entity_bonus += 0.3

            # Signal E: Structured table bonus for financial/tabular queries
            table_bonus = 0.15 if cand.get("metadata", {}).get("is_table") and any(w in q_lower for w in ["premium", "amount", "table", "sum", "insured", "schedule", "charge"]) else 0.0

            if ce_scores is not None:
                final_score = (0.70 * semantic_score) + (0.12 * exact_bonus) + (0.10 * term_density) + (0.04 * entity_bonus) + (0.04 * table_bonus)
            else:
                final_score = (0.40 * cand.get("score", 0.5)) + (0.35 * term_density) + (0.15 * exact_bonus) + (0.10 * entity_bonus)

            cand_copy["rerank_score"] = float(round(final_score, 4))
            scored_candidates.append((final_score, cand_copy))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored_candidates[:top_k]]


# Global Reranker Singleton
reranker = Reranker()
