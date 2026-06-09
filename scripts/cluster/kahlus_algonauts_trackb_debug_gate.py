#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_RUN_FILES = (
    "summary.json",
    "metrics.json",
    "metrics.csv",
    "eval_audit.json",
    "paper_mode_gate.json",
    "prepared_baseline_suite.json",
    "evidence_gate.json",
    "diagnostic_report.md",
    "RUN_REPORT.md",
    "EEG_MODEL_CARD.md",
    "leakage_demo.json",
    "identity_probe.json",
)

REQUIRED_PREPARED_FILES = (
    "data_manifest.json",
    "feature_manifest.json",
    "stimulus_manifest.json",
    "eval_audit.json",
    "event_manifest.json",
    "split_manifest.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply the Track B one-GPU Algonauts debug gate.")
    parser.add_argument("--root", required=True, help="Track B persistent root.")
    parser.add_argument("--run-dir", default=None, help="Debug run directory. Defaults to ROOT/runs/algonauts_pair_operator_debug_gate.")
    parser.add_argument("--prepared-root", default=None, help="Prepared Algonauts root with manifests.")
    parser.add_argument("--out", default=None, help="Output debug_gate.json path. Defaults to ROOT/debug_gate.json.")
    args = parser.parse_args()

    root = Path(args.root)
    run_dir = Path(args.run_dir) if args.run_dir else root / "runs" / "algonauts_pair_operator_debug_gate"
    prepared_root = Path(args.prepared_root) if args.prepared_root else None
    out = Path(args.out) if args.out else root / "debug_gate.json"

    payload = evaluate_debug_gate(root=root, run_dir=run_dir, prepared_root=prepared_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


def evaluate_debug_gate(*, root: Path, run_dir: Path, prepared_root: Path | None = None) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {
        "root": str(root),
        "run_dir": str(run_dir),
        "prepared_root": str(prepared_root) if prepared_root else None,
    }

    for name in REQUIRED_RUN_FILES:
        path = run_dir / name
        present = path.is_file() and path.stat().st_size > 0
        checks[f"run_file:{name}"] = present
        if not present:
            failures.append(f"required debug artifact missing or empty: {name}")

    if prepared_root is not None:
        for name in REQUIRED_PREPARED_FILES:
            path = prepared_root / name
            present = path.is_file() and path.stat().st_size > 0
            checks[f"prepared_file:{name}"] = present
            if not present:
                failures.append(f"required prepared artifact missing or empty: {name}")

    summary = _read_json(run_dir / "summary.json")
    metrics = _read_json(run_dir / "metrics.json")
    eval_audit = _read_json(run_dir / "eval_audit.json")
    paper_gate = _read_json(run_dir / "paper_mode_gate.json")
    suite = _read_json(run_dir / "prepared_baseline_suite.json")
    evidence_gate = _read_json(run_dir / "evidence_gate.json")
    leakage_demo = _read_json(run_dir / "leakage_demo.json")
    identity_probe = _read_json(run_dir / "identity_probe.json")

    if summary.get("status") != "completed_prepared_training":
        failures.append(f"summary status is not completed_prepared_training: {summary.get('status')}")
    completed_steps = _float_or_nan(summary.get("completed_steps"))
    if not math.isfinite(completed_steps) or completed_steps <= 0:
        failures.append(f"completed_steps is not positive/finite: {summary.get('completed_steps')}")
    if summary.get("quarantined_tasks"):
        failures.append(f"quarantined tasks present: {summary.get('quarantined_tasks')}")

    stimulus = summary.get("stimulus_evidence")
    checks["stimulus_evidence_claim_eligible"] = isinstance(stimulus, dict) and stimulus.get("claim_eligible") is True
    checks["stimulus_evidence_hash_verified"] = isinstance(stimulus, dict) and stimulus.get("hash_verified") is True
    feature_statuses = stimulus.get("feature_statuses") if isinstance(stimulus, dict) else None
    checks["stimulus_evidence_real_precomputed"] = isinstance(feature_statuses, list) and "real_precomputed" in feature_statuses
    if not checks["stimulus_evidence_claim_eligible"]:
        failures.append("stimulus_to_fmri_response is not claim-eligible in summary stimulus_evidence")
    if not checks["stimulus_evidence_hash_verified"]:
        failures.append("stimulus feature hashes were not verified in summary stimulus_evidence")
    if not checks["stimulus_evidence_real_precomputed"]:
        failures.append("stimulus features are not marked real_precomputed")

    task_row = _task_result(summary, "stimulus_to_fmri_response")
    if not task_row:
        failures.append("stimulus_to_fmri_response task result missing")
    elif task_row.get("status") not in (None, "completed"):
        failures.append(f"stimulus_to_fmri_response task did not complete: {task_row.get('status')}")
    for key in ("best_val_mse", "test_mse", "test_pearsonr"):
        if not _finite_number(task_row.get(key)):
            failures.append(f"stimulus_to_fmri_response {key} is missing/non-finite: {task_row.get(key)}")

    nonfinite_metrics = _nonfinite_number_paths(metrics)
    checks["metrics_nonfinite_paths"] = nonfinite_metrics
    if nonfinite_metrics:
        failures.append("metrics.json contains non-finite numbers: " + ", ".join(nonfinite_metrics[:20]))

    if eval_audit.get("passed") is not True:
        failures.append("eval_audit.json did not pass")
    if paper_gate.get("passed") is not True:
        failures.append("paper_mode_gate.json did not pass")

    stimulus_suite = (suite.get("tasks") or {}).get("stimulus_to_fmri_response") if isinstance(suite, dict) else None
    if not isinstance(stimulus_suite, dict):
        failures.append("prepared_baseline_suite missing stimulus_to_fmri_response task")
    else:
        metrics_by_model = stimulus_suite.get("metrics_by_model") or {}
        ranking = stimulus_suite.get("ranking") or []
        checks["baseline_models"] = sorted(metrics_by_model) if isinstance(metrics_by_model, dict) else []
        checks["baseline_ranking_rows"] = len(ranking) if isinstance(ranking, list) else 0
        for model_id in ("train_mean", "linear_ridge"):
            if not isinstance(metrics_by_model, dict) or model_id not in metrics_by_model:
                failures.append(f"prepared baseline model missing for debug gate: {model_id}")
        if not isinstance(ranking, list) or not ranking:
            failures.append("prepared baseline ranking missing for stimulus_to_fmri_response")

    evidence_failures = evidence_gate.get("failures") if isinstance(evidence_gate, dict) else None
    evidence_science_claim_only = evidence_gate.get("passed") is False and evidence_failures == [
        "summary.json scientific_claim_allowed is false"
    ]
    checks["scientific_evidence_gate_passed"] = evidence_gate.get("passed") is True
    checks["scientific_evidence_gate_failure_is_claim_flag_only"] = evidence_science_claim_only
    if evidence_gate.get("passed") is not True:
        if evidence_science_claim_only:
            warnings.append(
                "evidence_gate.json is false only because scientific_claim_allowed is disabled; this does not block the one-GPU debug spend gate"
            )
        else:
            failures.append(f"evidence_gate.json did not pass: {evidence_failures}")

    for name, payload in (("leakage_demo", leakage_demo), ("identity_probe", identity_probe)):
        observed = payload.get("observed_seeds") if isinstance(payload, dict) else None
        checks[f"{name}_observed_seeds"] = observed
        if observed != [0, 1, 2]:
            failures.append(f"{name}.json did not complete canonical seeds [0, 1, 2]: {observed}")

    return {
        "schema": "kahlus.trackb_debug_gate.v1",
        "passed": not failures,
        "debug_plumbing_passed": not failures,
        "scientific_evidence_gate_passed": checks["scientific_evidence_gate_passed"],
        "failures": failures,
        "warnings": warnings,
        "checks": checks,
        "notes": [
            "This gate controls whether it is safe to spend A100s on the ablation sweep.",
            "It is intentionally separate from scientific_evidence_gate_passed, which controls publication-claim eligibility.",
        ],
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {"_read_error": str(exc)}
    return payload if isinstance(payload, dict) else {"_payload_type": type(payload).__name__}


def _task_result(summary: dict[str, Any], task_id: str) -> dict[str, Any]:
    rows = summary.get("task_results")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and row.get("task_id") == task_id:
                return row
    if summary.get("task_id") == task_id:
        return summary
    return {}


def _nonfinite_number_paths(payload: Any, path: str = "") -> list[str]:
    out: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            out.extend(_nonfinite_number_paths(value, f"{path}.{key}" if path else str(key)))
    elif isinstance(payload, list):
        for idx, value in enumerate(payload):
            out.extend(_nonfinite_number_paths(value, f"{path}[{idx}]"))
    elif isinstance(payload, (int, float)) and not math.isfinite(float(payload)):
        out.append(path)
    return out


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
