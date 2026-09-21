"""
Motor Insurance Schema (Car, Two-Wheeler, Commercial Vehicle, Tractor).
"""

from typing import Optional, List
from pydantic import BaseModel, Field
from schemas.base import MotorSubtype


class MotorInsuranceSchema(BaseModel):
    subtype: MotorSubtype = MotorSubtype.CAR
    policy_number: Optional[str] = None
    policy_type: Optional[str] = None
    insurer: Optional[str] = None
    insured_name: Optional[str] = None
    insured_address: Optional[str] = None
    mobile: Optional[str] = None
    email: Optional[str] = None

    # Vehicle details
    registration_number: Optional[str] = None
    rto: Optional[str] = None
    vehicle_make: Optional[str] = None
    vehicle_model: Optional[str] = None
    vehicle_type: Optional[str] = None
    manufacturing_year: Optional[str] = None
    registration_date: Optional[str] = None
    engine_number: Optional[str] = None
    chassis_number: Optional[str] = None
    seating_capacity: Optional[str] = None
    cc: Optional[str] = None
    gvw: Optional[str] = None

    # Financial & Cover
    idv: Optional[float] = None
    own_damage_premium: Optional[float] = None
    third_party_premium: Optional[float] = None
    ncb_percentage: Optional[str] = None
    gst: Optional[float] = None
    total_premium: Optional[float] = None

    # Dates
    policy_start_date: Optional[str] = None
    policy_end_date: Optional[str] = None

    # Previous policy info
    previous_policy_number: Optional[str] = None
    previous_insurer: Optional[str] = None

    # Conditions & extras
    coverage: Optional[str] = None
    add_on_covers: List[str] = Field(default_factory=list)
    deductibles: Optional[str] = None
    limitations: Optional[str] = None
    driver_requirements: Optional[str] = None
    puc_requirement: Optional[str] = None
    nominee: Optional[str] = None
