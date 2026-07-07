from __future__ import annotations

from typing import Any, Mapping, Sequence

from neurotwin.gates.unified_gate import evaluate_gate

STF_CLAIM_SCOPE = "stf_epilepsy_benchmark_definition_ready"

REQUIRED_STF_TASKS: tuple[str, ...] = (
    "future_eeg_forecasting",
    "longer_horizon_eeg_forecasting",
    "held_out_channel_completion",
    "patient_held_out_event_risk_forecasting",
)

REQUIRED_STF_BASELINES_BY_TASK: dict[str, tuple[str, ...]] = {
    "future_eeg_forecasting": ("persistence", "ridge_ar", "tiny_ssm"),
    "longer_horizon_eeg_forecasting": ("persistence", "ridge_ar", "tiny_ssm"),
    "held_out_channel_completion": ("channel_mean", "ridge_ar", "tiny_ssm"),
    "patient_held_out_event_risk_forecasting": (
        "cycle_time_of_day",
        "event_frequency",
        "logistic_ridge",
    ),
}

REQUIRED_STF_NEGATIVE_CONTROLS: tuple[str, ...] = (
    "shuffled_target_control",
    "time_shifted_label_control",
)

REQUIRED_STF_SPLITS: tuple[str, ...] = (
    "patient_held_out",
    "time_held_out",
)

BLOCKED_STF_CLAIM_TERMS: tuple[str, ...] = (
    "diagnosis",
    "diagnostic",
    "treatment",
    "medication",
    "prevention",
    "prevents_seizures",
    "replaces_veeg",
    "replaces_psg",
    "clinical_predictor",
    "stimulation",
)


def stf_benchmark_contract() -> dict[str, Any]:
    """Return the minimum STF benchmark definition.

    This is a contract, not a runner. Data loading and model training belong in a later
    sprint after dataset selection and another literature/math stop gate.
    """

    return {
        "schema": "kahlus.stf_benchmark_contract.v1",
        "claim_scope": STF_CLAIM_SCOPE,
        "required_tasks": list(REQUIRED_STF_TASKS),
        "required_baselines_by_task": {
            task_id: list(baselines) for task_id, baselines in REQUIRED_STF_BASELINES_BY_TASK.items()
        },
        "required_negative_controls": list(REQUIRED_STF_NEGATIVE_CONTROLS),
        "required_splits": list(REQUIRED_STF_SPLITS),
        "blocked_claim_terms": list(BLOCKED_STF_CLAIM_TERMS),
        "implementation_boundary": (
            "passive benchmark definition only; no diagnosis, treatment, stimulation, "
            "medication, wearable-device efficacy, or A100 claim"
        ),
    }


def audit_stf_benchmark_contract(
    *,
    declared_tasks: Sequence[str],
    baselines_by_task: Mapping[str, Sequence[str]],
    negative_controls: Sequence[str],
    split_types: Sequence[str],
    claim_scope: str = STF_CLAIM_SCOPE,
) -> dict[str, Any]:
    """Audit a proposed STF benchmark definition against the minimum contract."""

    task_set = {str(task) for task in declared_tasks}
    control_set = {str(control) for control in negative_controls}
    split_set = {str(split) for split in split_types}
    baseline_sets = {
        str(task): {str(baseline) for baseline in baselines}
        for task, baselines in baselines_by_task.items()
    }

    failures: list[str] = []
    for task_id in REQUIRED_STF_TASKS:
        if task_id not in task_set:
            failures.append(f"required task missing: {task_id}")
    for task_id, required_baselines in REQUIRED_STF_BASELINES_BY_TASK.items():
        present = baseline_sets.get(task_id, set())
        for baseline_id in required_baselines:
            if baseline_id not in present:
                failures.append(f"required baseline missing for {task_id}: {baseline_id}")
    for control_id in REQUIRED_STF_NEGATIVE_CONTROLS:
        if control_id not in control_set:
            failures.append(f"required negative control missing: {control_id}")
    for split_id in REQUIRED_STF_SPLITS:
        if split_id not in split_set:
            failures.append(f"required split missing: {split_id}")
    if any(term in str(claim_scope).lower() for term in BLOCKED_STF_CLAIM_TERMS):
        failures.append(f"blocked clinical/device claim term in scope: {claim_scope!r}")

    return {
        "schema": "kahlus.stf_benchmark_contract_audit.v1",
        "passed": not failures,
        "failure_reasons": failures,
        "declared_tasks": sorted(task_set),
        "declared_negative_controls": sorted(control_set),
        "declared_split_types": sorted(split_set),
        "claim_scope": str(claim_scope),
    }


def build_stf_gate(
    *,
    dataset: str,
    declared_tasks: Sequence[str],
    baselines_by_task: Mapping[str, Sequence[str]],
    negative_controls: Sequence[str],
    split_types: Sequence[str],
    split_audit_passed: bool,
    baseline_table_present: bool,
    finite_metrics: bool,
    calibration_checked: bool,
    claim_scope: str = STF_CLAIM_SCOPE,
    extra_failure_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the narrow STF evidence gate for benchmark-definition readiness."""

    contract_audit = audit_stf_benchmark_contract(
        declared_tasks=declared_tasks,
        baselines_by_task=baselines_by_task,
        negative_controls=negative_controls,
        split_types=split_types,
        claim_scope=claim_scope,
    )
    failures = list(contract_audit["failure_reasons"])
    failures.extend(str(reason) for reason in extra_failure_reasons if str(reason).strip())

    gate = evaluate_gate(
        branch="stf",
        dataset=dataset,
        split_audit_passed=split_audit_passed,
        baseline_table_present=baseline_table_present,
        finite_metrics=finite_metrics,
        calibration_checked=calibration_checked,
        claim_scope=claim_scope,
        extra_failure_reasons=failures,
    )
    gate["gate_criteria"] = {
        "allowed_claim_scope": STF_CLAIM_SCOPE,
        "required_tasks": list(REQUIRED_STF_TASKS),
        "required_baselines_by_task": {
            task_id: list(baselines)
            for task_id, baselines in REQUIRED_STF_BASELINES_BY_TASK.items()
        },
        "required_negative_controls": list(REQUIRED_STF_NEGATIVE_CONTROLS),
        "required_splits": list(REQUIRED_STF_SPLITS),
        "blocked_claim_terms": list(BLOCKED_STF_CLAIM_TERMS),
        "requires_split_audit_passed": True,
        "requires_baseline_table_present": True,
        "requires_finite_metrics": True,
        "requires_calibration_checked": True,
    }
    gate["benchmark_contract_audit"] = contract_audit
    return gate
