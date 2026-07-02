from __future__ import annotations

from typing import Any, Sequence

from neurotwin.gates import evaluate_gate

EEG_V1_CLAIM_SCOPE = "eeg_future_forecasting_benchmark_ready"
EEG_V1_ADAPTATION_CLAIM_SCOPE = "eeg_fewshot_adaptation_benchmark_ready"
_ALLOWED_SPLITS = frozenset({"subject_held_out", "session_held_out"})


def build_eeg_v1_gate(
    *,
    dataset: str,
    split_audit_passed: bool,
    baseline_table_present: bool,
    finite_metrics: bool,
    forecast_horizon: int,
    split_type: str,
    claim_scope: str = EEG_V1_CLAIM_SCOPE,
    extra_failure_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    """Gate the narrow v1 benchmark-readiness claim."""

    extra: list[str] = []
    if int(forecast_horizon) < 1:
        extra.append("forecast horizon not declared")
    if split_type not in _ALLOWED_SPLITS:
        extra.append("subject-held-out or session-held-out split not declared")
    extra.extend(str(reason) for reason in extra_failure_reasons if str(reason).strip())
    gate = evaluate_gate(
        branch="v1",
        dataset=dataset,
        split_audit_passed=split_audit_passed,
        baseline_table_present=baseline_table_present,
        finite_metrics=finite_metrics,
        calibration_checked=True,
        claim_scope=claim_scope,
        extra_failure_reasons=extra,
    )
    gate["gate_criteria"] = {
        "min_forecast_horizon": 1,
        "allowed_split_types": sorted(_ALLOWED_SPLITS),
        "requires_split_audit_passed": True,
        "requires_baseline_table_present": True,
        "requires_finite_metrics": True,
        "requires_calibration_checked": True,
        "allowed_claim_scope": EEG_V1_CLAIM_SCOPE,
    }
    return gate


def build_eeg_v1_adaptation_gate(
    *,
    dataset: str,
    split_audit_passed: bool,
    baseline_table_present: bool,
    finite_metrics: bool,
    support_windows: int,
    query_windows: int,
    claim_scope: str = EEG_V1_ADAPTATION_CLAIM_SCOPE,
) -> dict[str, Any]:
    """Gate the narrow v1 few-shot adaptation benchmark-readiness claim."""

    extra: list[str] = []
    if int(support_windows) < 1:
        extra.append("support windows not declared")
    if int(query_windows) < 1:
        extra.append("query windows not declared")
    gate = evaluate_gate(
        branch="v1",
        dataset=dataset,
        split_audit_passed=split_audit_passed,
        baseline_table_present=baseline_table_present,
        finite_metrics=finite_metrics,
        calibration_checked=True,
        claim_scope=claim_scope,
        extra_failure_reasons=extra,
    )
    gate["gate_criteria"] = {
        "min_support_windows": 1,
        "min_query_windows": 1,
        "requires_split_audit_passed": True,
        "requires_baseline_table_present": True,
        "requires_finite_metrics": True,
        "requires_calibration_checked": True,
        "allowed_claim_scope": EEG_V1_ADAPTATION_CLAIM_SCOPE,
    }
    return gate
