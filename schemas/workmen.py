"""
Workmen Compensation Insurance Schema.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class WorkmenCompensationSchema(BaseModel):
    policy_number: Optional[str] = None
    employer: Optional[str] = None
    business_activity: Optional[str] = None
    nature_of_work: Optional[str] = None
    address: Optional[str] = None
    gst: Optional[str] = None
    insurer: Optional[str] = None

    policy_start_date: Optional[str] = None
    policy_end_date: Optional[str] = None

    territory: Optional[str] = None
    jurisdiction: Optional[str] = None
    sum_insured: Optional[float] = None
    employee_details: Optional[str] = None
    number_of_employees: Optional[int] = None
    employment_location: Optional[str] = None

    medical_expenses_limit: Optional[str] = None
    occupational_disease_cover: Optional[str] = None
    contractor_cover: Optional[str] = None
    subcontractor_cover: Optional[str] = None

    extensions: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    warranties: List[str] = Field(default_factory=list)
    exclusions: List[str] = Field(default_factory=list)

    premium: Optional[float] = None
    gst: Optional[float] = None
    total_premium: Optional[float] = None
