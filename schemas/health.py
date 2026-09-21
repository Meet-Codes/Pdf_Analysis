"""
Health Insurance Schema.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class HealthMember(BaseModel):
    name: Optional[str] = None
    date_of_birth: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    member_id: Optional[str] = None
    relationship: Optional[str] = None
    inception_date: Optional[str] = None
    pre_existing_disease: Optional[str] = None
    critical_illness_cover: Optional[str] = None
    personal_accident_cover: Optional[str] = None


class HealthInsuranceSchema(BaseModel):
    policy_number: Optional[str] = None
    proposal_number: Optional[str] = None
    customer_id: Optional[str] = None
    insurer: Optional[str] = None
    policyholder_name: Optional[str] = None
    address: Optional[str] = None
    mobile: Optional[str] = None
    email: Optional[str] = None

    product_name: Optional[str] = None
    plan_name: Optional[str] = None
    policy_type: Optional[str] = None

    policy_start_date: Optional[str] = None
    policy_end_date: Optional[str] = None

    sum_insured: Optional[float] = None
    cumulative_bonus: Optional[float] = None
    deductible: Optional[str] = None

    family_members: List[HealthMember] = Field(default_factory=list)

    nominee: Optional[str] = None
    portability: Optional[str] = None
    previous_policy: Optional[str] = None

    premium: Optional[float] = None
    gst: Optional[float] = None
    discount: Optional[float] = None
    net_premium: Optional[float] = None

    hospitalization_cover: Optional[str] = None
    pre_hospitalization: Optional[str] = None
    post_hospitalization: Optional[str] = None
    daycare: Optional[str] = None
    home_care: Optional[str] = None
    hospital_cash: Optional[str] = None

    waiting_periods: Optional[str] = None
    exclusions: List[str] = Field(default_factory=list)
    claim_documents: Optional[str] = None
    network_hospitals: Optional[str] = None
    renewal_conditions: Optional[str] = None
