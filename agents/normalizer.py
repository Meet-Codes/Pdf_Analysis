"""
Normalizer Agent: Normalizes raw extracted fields into clean canonical representation.
Arbitrates among competing extraction paths via CandidateResolver.
Enforces zero-hallucination fail-closed policy and returns strictly JSON-serializable primitives.
"""

from typing import Dict, Any, Tuple, Optional
from schemas.base import DocumentType, SourceEvidence
from retrieval.candidate_resolver import CandidateResolver
from utils.logger import get_logger

logger = get_logger("normalizer")


def normalize_document_content(
    raw_data: Dict[str, Any],
    doc_type: DocumentType,
    page_texts: Optional[Dict[int, str]] = None,
    page_sources: Optional[Dict[int, str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, SourceEvidence]]:
    """
    Normalizes all extracted fields using CandidateResolver.
    Arbitrates between dedicated resolvers, table cells, spatial coordinates,
    deterministic regex, and LLMs to select exactly ONE grounded canonical candidate per field.
    Guarantees that normalized_data contains strictly JSON-serializable primitives and
    ZERO non-serializable objects (no _evidence_candidates, no ResolvedFieldCandidate).
    Returns: (canonical_structured_data, evidence_map)
    """
    resolver = CandidateResolver(doc_type=doc_type)
    resolver.load_from_raw_extraction(raw_data=raw_data)
    canonical_data, evidence_map = resolver.resolve_all(page_texts=page_texts)

    logger.info(
        f"Normalized {len([v for v in canonical_data.values() if v is not None])} canonical fields "
        f"for {doc_type.value} with {len(evidence_map)} grounded evidence records."
    )

    return canonical_data, evidence_map
