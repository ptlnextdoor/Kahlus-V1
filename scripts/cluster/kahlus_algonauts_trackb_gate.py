#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


STRICT_DELTA = 0.01
RIDGE_TOLERANCE = 0.01
SEEDS = (0, 1, 2)
MODEL_ARMS = (
    "current_neurotwin",
    "pair_operator_no_pair_state",
    "pair_operator_low_rank_pair_state",
    "pair_operator_pair_state_uncertainty",
    "pair_operator_full",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply the Track B strict sweep gate.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    payload = evaluate_gate(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


def evaluate_gate(root: Path) -> dict[str, Any]:
    metrics = _collect_metrics(root)
    failures: list[str] = []
    missing = [
        f"{arm}:seed{seed}"
        for arm in ("ridge_anchor", *MODEL_ARMS)
        for seed in SEEDS
        if not math.isfinite(metrics.get(arm, {}).get(str(seed), {}).get("stimulus_to_fmri_response_pearsonr", float("nan")))
    ]
    if missing:
        failures.append("missing stimulus_to_fmri_response Pearson metrics: " + ", ".join(missing[:20]))
    means = {
        arm: _finite_mean(row.get(str(seed), {}).get("stimulus_to_fmri_response_pearsonr") for seed in SEEDS)
        for arm, row in metrics.items()
    }
    full = means.get("pair_operator_full")
    no_pair = means.get("pair_operator_no_pair_state")
    current = means.get("current_neurotwin")
    ridge = means.get("ridge_anchor")
    uncertainty_corr = _finite_mean(
        metrics.get("pair_operator_pair_state_uncertainty", {}).get(str(seed), {}).get("uncertainty_error_spearman")
        for seed in SEEDS
    )
    if full is None or no_pair is None or full - no_pair < STRICT_DELTA:
        failures.append(
            f"pair_operator_full did not beat no_pair by >= {STRICT_DELTA}: "
            f"full={full} no_pair={no_pair}"
        )
    if full is None or current is None or full - current < STRICT_DELTA:
        failures.append(
            f"pair_operator_full did not beat current_neurotwin by >= {STRICT_DELTA}: "
            f"full={full} current={current}"
        )
    if full is None or ridge is None or full < ridge - RIDGE_TOLERANCE:
        failures.append(
            f"pair_operator_full is below ridge anchor tolerance: full={full} ridge={ridge} tolerance={RIDGE_TOLERANCE}"
        )
    positive_no_pair = _positive_seed_count(metrics, "pair_operator_full", "pair_operator_no_pair_state", STRICT_DELTA)
    positive_current = _positive_seed_count(metrics, "pair_operator_full", "current_neurotwin", STRICT_DELTA)
    if positive_no_pair < 2:
        failures.append(f"full > no_pair by threshold in only {positive_no_pair}/3 seeds")
    if positive_current < 2:
        failures.append(f"full > current by threshold in only {positive_current}/3 seeds")
    if uncertainty_corr is None or uncertainty_corr <= 0:
        failures.append(f"uncertainty-error correlation is not positive/finite: {uncertainty_corr}")
    quarantined = _quarantined_tasks(root)
    if quarantined:
        failures.append("quarantined required tasks: " + ", ".join(quarantined[:20]))
    return {
        "passed": not failures,
        "failures": failures,
        "strict_delta": STRICT_DELTA,
        "ridge_tolerance": RIDGE_TOLERANCE,
        "metrics": metrics,
        "mean_pearsonr": means,
        "positive_seed_counts": {
            "full_vs_no_pair": positive_no_pair,
            "full_vs_current": positive_current,
        },
        "uncertainty_error_spearman_mean": uncertainty_corr,
        "notes": [
            "This gate is intentionally strict and blocks the long run on missing metrics.",
            "The CI and subject-consistency checks require richer per-subject artifacts; absence should be treated as a failed reviewer gate.",
        ],
    }


def _collect_metrics(root: Path) -> dict[str, dict[str, dict[str, float]]]:
    metrics: dict[str, dict[str, dict[str, float]]] = {"ridge_anchor": {}}
    for arm in MODEL_ARMS:
        metrics[arm] = {}
    for seed in SEEDS:
        metrics["ridge_anchor"][str(seed)] = _ridge_metrics(root / "sweep" / f"seed{seed}" / "ridge_anchor" / "prepared_baseline_suite.json")
        for arm in MODEL_ARMS:
            metrics[arm][str(seed)] = _summary_metrics(root / "runs" / f"{arm}_seed{seed}" / "summary.json")
    return metrics


def _ridge_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    task = payload.get("tasks", {}).get("stimulus_to_fmri_response", {})
    by_model = task.get("metrics_by_model", {}) if isinstance(task, dict) else {}
    ridge = by_model.get("linear_ridge") or by_model.get("autoregressive_ridge") or {}
    return {
        "stimulus_to_fmri_response_pearsonr": _float_or_nan(ridge.get("pearsonr")),
        "stimulus_to_fmri_response_mse": _float_or_nan(ridge.get("mse")),
    }


def _summary_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    task_result = _task_result(payload, "stimulus_to_fmri_response")
    if not task_result:
        return {}
    return {
        "stimulus_to_fmri_response_pearsonr": _float_or_nan(task_result.get("test_pearsonr", task_result.get("eval_pearsonr"))),
        "stimulus_to_fmri_response_mse": _float_or_nan(task_result.get("test_mse", task_result.get("eval_mse"))),
        "uncertainty_error_spearman": _float_or_nan(
            task_result.get("uncertainty_error_spearman", task_result.get("error_uncertainty_spearman"))
        ),
    }


def _task_result(payload: dict[str, Any], task_id: str) -> dict[str, Any]:
    for row in payload.get("task_results", []):
        if isinstance(row, dict) and row.get("task_id") == task_id:
            return row
    if payload.get("task_id") == task_id:
        return payload
    return {}


def _quarantined_tasks(root: Path) -> list[str]:
    out: list[str] = []
    for path in (root / "runs").glob("*_seed*/summary.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            out.append(f"{path}: invalid summary json")
            continue
        for row in payload.get("quarantined_tasks", []):
            if isinstance(row, dict):
                out.append(f"{path.parent.name}:{row.get('task_id')}:{row.get('reason')}")
    return out


def _positive_seed_count(metrics: dict[str, dict[str, dict[str, float]]], left: str, right: str, threshold: float) -> int:
    count = 0
    for seed in SEEDS:
        left_value = metrics.get(left, {}).get(str(seed), {}).get("stimulus_to_fmri_response_pearsonr", float("nan"))
        right_value = metrics.get(right, {}).get(str(seed), {}).get("stimulus_to_fmri_response_pearsonr", float("nan"))
        if math.isfinite(left_value) and math.isfinite(right_value) and left_value - right_value >= threshold:
            count += 1
    return count


def _finite_mean(values: Any) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return mean(finite) if finite else None


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
