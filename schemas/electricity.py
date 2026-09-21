"""
Electricity Bill Schema.
"""

from typing import Optional
from pydantic import BaseModel


class ElectricityBillSchema(BaseModel):
    electricity_provider: Optional[str] = None
    consumer_number: Optional[str] = None
    customer_name: Optional[str] = None
    address: Optional[str] = None
    meter_number: Optional[str] = None
    phase: Optional[str] = None
    tariff: Optional[str] = None

    bill_number: Optional[str] = None
    bill_date: Optional[str] = None
    due_date: Optional[str] = None

    previous_reading: Optional[float] = None
    current_reading: Optional[float] = None
    consumption: Optional[float] = None
    meter_multiplier: Optional[float] = None
    total_consumption: Optional[float] = None

    fixed_charge: Optional[float] = None
    energy_charge: Optional[float] = None
    fuel_charge: Optional[float] = None
    electricity_duty: Optional[float] = None
    meter_charge: Optional[float] = None
    other_charges: Optional[float] = None
    arrears: Optional[float] = None
    adjustments: Optional[float] = None
    interest: Optional[float] = None

    total_bill: Optional[float] = None
    net_bill_amount: Optional[float] = None
    previous_month_usage: Optional[float] = None
    previous_bill_amount: Optional[float] = None
