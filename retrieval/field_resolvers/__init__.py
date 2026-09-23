"""
Field Resolvers: Dedicated, deterministic extraction modules for core document entities.
"""

from retrieval.field_resolvers.base_resolver import BaseFieldResolver, ResolvedFieldCandidate
from retrieval.field_resolvers.policy_number_resolver import PolicyNumberResolver
from retrieval.field_resolvers.customer_name_resolver import CustomerNameResolver
from retrieval.field_resolvers.date_resolver import DateResolver
from retrieval.field_resolvers.amount_resolver import AmountResolver
from retrieval.field_resolvers.insurer_resolver import InsurerResolver
from retrieval.field_resolvers.vehicle_resolver import VehicleResolver

__all__ = [
    "BaseFieldResolver",
    "ResolvedFieldCandidate",
    "PolicyNumberResolver",
    "CustomerNameResolver",
    "DateResolver",
    "AmountResolver",
    "InsurerResolver",
    "VehicleResolver",
]
