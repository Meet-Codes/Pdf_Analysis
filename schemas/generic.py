"""
Generic Document Schema for general documents, invoices, legal notices, and agreements.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class GenericDocumentSchema(BaseModel):
    document_title: Optional[str] = None
    issuing_organization: Optional[str] = None
    reference_number: Optional[str] = None
    primary_party_name: Optional[str] = None
    secondary_party_name: Optional[str] = None
    date_of_issue: Optional[str] = None
    expiry_or_due_date: Optional[str] = None
    amount_or_value: Optional[float] = None
    summary_points: List[str] = Field(default_factory=list)
    key_fields: Dict[str, Any] = Field(default_factory=dict)
