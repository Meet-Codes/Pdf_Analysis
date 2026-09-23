"""
Base Field Resolver: Core abstractions, candidate representation, and verification logic.
"""

from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field


class ResolvedFieldCandidate(BaseModel):
    field_name: str
    value: Any
    exact_label: Optional[str] = None
    raw_evidence: str = ""
    page: int = 1
    bbox: Optional[Tuple[float, float, float, float]] = None
    confidence: float = 0.0
    method: str = "deterministic"
    source: str = "native"
    is_verified: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_display_string(self) -> str:
        s = str(self.value)
        if self.page:
            return f"{s}\n\nSource: Page {self.page}"
        return s


class BaseFieldResolver:
    """Base class for dedicated field resolvers."""
    field_name: str = "base_field"
    aliases: List[str] = []

    def resolve(
        self,
        full_text: str,
        page_texts: Dict[int, str],
        layout: Optional[Any] = None,
        tables: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[ResolvedFieldCandidate]:
        raise NotImplementedError("Subclasses must implement resolve()")
