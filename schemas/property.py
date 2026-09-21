"""
Property & Fire Insurance Schema.
"""

from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class RiskLocation(BaseModel):
    address: Optional[str] = None
    occupancy: Optional[str] = None
    property_description: Optional[str] = None


class SumInsuredBreakdown(BaseModel):
    building: Optional[float] = None
    plant_machinery: Optional[float] = None
    furniture: Optional[float] = None
    equipment: Optional[float] = None
    stock: Optional[float] = None
    other: Optional[float] = None
    total: Optional[float] = None


class PropertyInsuranceSchema(BaseModel):
    policy_number: Optional[str] = None
    insured_business: Optional[str] = None
    business_type: Optional[str] = None
    gst: Optional[str] = None
    address: Optional[str] = None
    insurer: Optional[str] = None

    policy_start_date: Optional[str] = None
    policy_end_date: Optional[str] = None

    risk_locations: List[RiskLocation] = Field(default_factory=list)
    sum_insured_breakdown: Optional[SumInsuredBreakdown] = None
    total_sum_insured: Optional[float] = None

    base_premium: Optional[float] = None
    terrorism_premium: Optional[float] = None
    gst_amount: Optional[float] = None
    total_premium: Optional[float] = None

    earthquake_cover: Optional[str] = None
    terrorism_cover: Optional[str] = None
    clauses: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    warranties: List[str] = Field(default_factory=list)
    exclusions: List[str] = Field(default_factory=list)
    deductibles: Optional[str] = None
