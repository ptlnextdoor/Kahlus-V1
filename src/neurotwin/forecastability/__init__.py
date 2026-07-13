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
from neurotwin.forecastability.hnph_gate import (
    HNPH_BASELINE_FEASIBILITY_SCHEMA,
    HnphFeasibilityEvidence,
    build_hnph_baseline_feasibility,
    format_hnph_baseline_feasibility,
    run_hnph_baseline_feasibility,
)
from neurotwin.forecastability.hnph_baselines import (
    HNPH_CLASSICAL_BASELINE_SCHEMA,
    HNPH_PRIMARY_SOFT_OUTCOME_COUNT,
    HnphBaselineError,
    HnphClassicalBaselineResult,
    HnphClassicalTrial,
    run_hnph_classical_baselines,
    write_hnph_classical_baselines,
)
from neurotwin.forecastability.label_reliability import (
    HNPH_PRIMARY_OUTCOME_ALPHABET,
    LABEL_REPRODUCIBILITY_FAMILY_SCHEMA,
    LabelReliabilityError,
    LabelReproducibilityBandInput,
    LabelReproducibilityFamilyResult,
    LabelReproducibilityReference,
    build_leave_one_rater_out_soft_targets,
    estimate_label_reproducibility_family,
    estimate_label_reproducibility_reference,
    relative_to_label_reproducibility,
)
from neurotwin.forecastability.transition_targets import (
    TransitionEpoch,
    TransitionLeadBand,
    TransitionTarget,
    TransitionTargetBuildResult,
    TransitionTargetSpec,
    build_natural_transition_target_result,
    build_natural_transition_targets,
)

__all__ = [
    "CANONICAL_PHYSICAL_UNITS",
    "OUTCOME_CLASSES",
    "QUALITY_STATES",
    "EvidenceDecision",
    "FORBIDDEN_MODEL_INPUT_FIELDS",
    "CausalPreprocessingSpec",
    "FirebreakAuditReport",
    "ForecastAnchor",
    "HNPH_BASELINE_FEASIBILITY_SCHEMA",
    "HnphFeasibilityEvidence",
    "HNPH_CLASSICAL_BASELINE_SCHEMA",
    "HNPH_PRIMARY_OUTCOME_ALPHABET",
    "HNPH_PRIMARY_SOFT_OUTCOME_COUNT",
    "LABEL_REPRODUCIBILITY_FAMILY_SCHEMA",
    "HnphBaselineError",
    "HnphClassicalBaselineResult",
    "HnphClassicalTrial",
    "LabelReliabilityError",
    "LabelReproducibilityBandInput",
    "LabelReproducibilityFamilyResult",
    "LabelReproducibilityReference",
    "LeadGeometry",
    "PhysicalRecordRegistry",
    "PhysicalSignalRecord",
    "QualityInterval",
    "StateTargetSpec",
    "TransitionEpoch",
    "TransitionLeadBand",
    "TransitionTarget",
    "TransitionTargetBuildResult",
    "TransitionTargetSpec",
    "TimeInterval",
    "audit_forecast_firebreak",
    "audit_model_input_fields",
    "audit_model_input_mapping",
    "build_hnph_baseline_feasibility",
    "build_leave_one_rater_out_soft_targets",
    "build_natural_transition_target_result",
    "build_natural_transition_targets",
    "format_hnph_baseline_feasibility",
    "estimate_label_reproducibility_reference",
    "estimate_label_reproducibility_family",
    "relative_to_label_reproducibility",
    "run_m0_gate",
    "run_m1_gate",
    "run_m2_gate",
    "run_m3_gate",
    "run_m4_gate",
    "run_m5_gate",
    "run_hnph_baseline_feasibility",
    "run_hnph_classical_baselines",
    "write_hnph_classical_baselines",
]
