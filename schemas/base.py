"""
Base schemas, enums, and canonical document representations.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    MOTOR_INSURANCE = "MOTOR_INSURANCE"
    HEALTH_INSURANCE = "HEALTH_INSURANCE"
    PROPERTY_INSURANCE = "PROPERTY_INSURANCE"
    WORKMEN_COMPENSATION = "WORKMEN_COMPENSATION"
    ELECTRICITY_BILL = "ELECTRICITY_BILL"
    INVOICE = "INVOICE"
    IDENTITY_DOCUMENT = "IDENTITY_DOCUMENT"
    LEGAL_DOCUMENT = "LEGAL_DOCUMENT"
    FINANCIAL_DOCUMENT = "FINANCIAL_DOCUMENT"
    EDUCATIONAL_DOCUMENT = "EDUCATIONAL_DOCUMENT"
    GENERIC_DOCUMENT = "GENERIC_DOCUMENT"


class MotorSubtype(str, Enum):
    CAR = "CAR"
    TWO_WHEELER = "TWO_WHEELER"
    COMMERCIAL_VEHICLE = "COMMERCIAL_VEHICLE"
    TRACTOR = "TRACTOR"
    GENERIC_MOTOR = "GENERIC_MOTOR"


class PageType(str, Enum):
    TEXT = "TEXT"
    SCANNED_IMAGE = "SCANNED_IMAGE"
    IMAGE = "IMAGE"
    TEXT_AND_IMAGE = "TEXT_AND_IMAGE"
    TABLE = "TABLE"
    FORM = "FORM"
    SIGNATURE = "SIGNATURE"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


class SourceEvidence(BaseModel):
    field: str
    value: Any
    page: int = 1
    source: str = "text"  # text, ocr, table, form
    evidence: str = ""


class PageInspectionResult(BaseModel):
    page_number: int
    has_text: bool
    text_length: int = 0
    image_count: int = 0
    image_coverage_ratio: float = 0.0
    likely_scanned: bool = False
    likely_table: bool = False
    likely_form: bool = False
    has_digital_signature: bool = False
    requires_ocr: bool = False
    page_type: PageType = PageType.UNKNOWN
    width: float = 0.0
    height: float = 0.0


class DocumentInspectionResult(BaseModel):
    document_id: str
    file_name: str
    file_size_bytes: int
    page_count: int
    is_encrypted: bool = False
    requires_password: bool = False
    can_copy: bool = True
    can_print: bool = True
    can_modify: bool = True
    can_annotate: bool = True
    has_embedded_files: bool = False
    embedded_file_names: List[str] = Field(default_factory=list)
    has_digital_signatures: bool = False
    detected_pdf_categories: List[str] = Field(default_factory=list)
    overall_requires_ocr: bool = False
    pages: List[PageInspectionResult] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseExtractedData(BaseModel):
    document_type: DocumentType = DocumentType.GENERIC_DOCUMENT
    raw_fields: Dict[str, Any] = Field(default_factory=dict)


class CanonicalDocument(BaseModel):
    document_id: str
    file_name: str
    document_type: DocumentType
    document_subtype: Optional[str] = None
    title: str = "Document Intelligence"
    summary: str = ""
    # Structured normalized data dict for serialization & export
    structured_data: Dict[str, Any] = Field(default_factory=dict)
    # Visual sections for clean production UI display
    sections: List[Dict[str, Any]] = Field(default_factory=list)
    # Field-level source evidence (hidden in normal UI, visible in Debug Mode)
    evidence: Dict[str, SourceEvidence] = Field(default_factory=dict)
    page_texts: Dict[int, str] = Field(default_factory=dict)
    # Validation results
    is_valid: bool = True
    validation_warnings: List[str] = Field(default_factory=list)
    validation_errors: List[str] = Field(default_factory=list)
    inspection: Optional[DocumentInspectionResult] = None
