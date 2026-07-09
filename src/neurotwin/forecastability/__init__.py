"""Forecastability benchmark gates and HNPH physical-record contracts."""

from neurotwin.forecastability.contracts import (
    CANONICAL_PHYSICAL_UNITS,
    OUTCOME_CLASSES,
    QUALITY_STATES,
    EvidenceDecision,
    LeadGeometry,
    PhysicalSignalRecord,
    QualityInterval,
    StateTargetSpec,
)
from neurotwin.forecastability.firebreak import (
    FORBIDDEN_MODEL_INPUT_FIELDS,
    CausalPreprocessingSpec,
    FirebreakAuditReport,
    ForecastAnchor,
    TimeInterval,
    audit_forecast_firebreak,
    audit_model_input_fields,
    audit_model_input_mapping,
)
from neurotwin.forecastability.m0 import run_m0_gate
from neurotwin.forecastability.m1 import run_m1_gate
from neurotwin.forecastability.m2 import run_m2_gate
from neurotwin.forecastability.m3 import run_m3_gate
from neurotwin.forecastability.m4 import run_m4_gate
from neurotwin.forecastability.m5 import run_m5_gate
from neurotwin.forecastability.registry import PhysicalRecordRegistry

__all__ = [
    "CANONICAL_PHYSICAL_UNITS",
    "OUTCOME_CLASSES",
    "QUALITY_STATES",
    "EvidenceDecision",
    "FORBIDDEN_MODEL_INPUT_FIELDS",
    "CausalPreprocessingSpec",
    "FirebreakAuditReport",
    "ForecastAnchor",
    "LeadGeometry",
    "PhysicalRecordRegistry",
    "PhysicalSignalRecord",
    "QualityInterval",
    "StateTargetSpec",
    "TimeInterval",
    "audit_forecast_firebreak",
    "audit_model_input_fields",
    "audit_model_input_mapping",
    "run_m0_gate",
    "run_m1_gate",
    "run_m2_gate",
    "run_m3_gate",
    "run_m4_gate",
    "run_m5_gate",
]
