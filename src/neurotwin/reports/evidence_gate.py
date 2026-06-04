from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from neurotwin.repro import write_json


def write_final_prepared_evidence_gate(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir)
    if not path.exists():
        raise ValueError(f"evidence-gate run-dir does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"evidence-gate run-dir is not a directory: {path}")
    summary = read_json_artifact(path / "summary.json")
    if not isinstance(summary, dict) or not summary:
        summary = {"scientific_claim_allowed": False, "status": "missing_summary"}
    eval_audit = read_json_artifact(path / "eval_audit.json")
    evidence_gate = build_prepared_evidence_gate(
        summary,
        eval_audit=eval_audit if isinstance(eval_audit, dict) else None,
        run_dir=path,
        stage="final",
    )
    write_json(path / "evidence_gate.json", evidence_gate)
    (path / "diagnostic_report.md").write_text(format_evidence_diagnostic_report(summary, evidence_gate), encoding="utf-8")
    return evidence_gate


def build_prepared_evidence_gate(
    summary: dict[str, Any],
    eval_audit: dict[str, Any] | None = None,
    *,
    run_dir: str | Path | None = None,
    stage: str = "final",
) -> dict[str, Any]:
    run_path = Path(run_dir) if run_dir is not None else None
    failures: list[str] = []
    if not isinstance(eval_audit, dict) or not eval_audit:
        failures.append("eval_audit.json missing")
    elif not bool(eval_audit.get("passed")):
        failures.append("eval audit did not pass")
    quarantined = summary.get("quarantined_tasks")
    if isinstance(quarantined, (list, tuple)) and quarantined:
        failed_tasks = ",".join(str(row.get("task_id", "unknown")) for row in quarantined if isinstance(row, dict))
        failures.append(f"required task quarantined: {failed_tasks or 'unknown'}")
    for row in _task_result_rows(summary):
        if row.get("test_mse") is None or row.get("best_val_mse") is None:
            failures.append(f"required task has missing/non-finite selected metric: {row.get('task_id', 'unknown')}")
    stimulus = summary.get("stimulus_evidence")
    if isinstance(stimulus, dict) and stimulus and not bool(stimulus.get("claim_eligible")):
        failures.append("stimulus-to-fMRI evidence is not claim eligible")
    baseline_ranking_present = _baseline_ranking_present(run_path)
    competitor_reproduction_status_present = _competitor_reproduction_status_present(run_path)
    paper_mode_gate_present = _paper_mode_gate_present(run_path)
    if not baseline_ranking_present:
        failures.append("baseline ranking artifact missing or unavailable")
    if not competitor_reproduction_status_present:
        failures.append("exact competitor reproduction status requires prepared baseline suite artifacts")
    if not paper_mode_gate_present:
        failures.append("paper_mode_gate.json missing or not passed")
    summary_claim = bool(summary.get("scientific_claim_allowed"))
    if not summary_claim:
        failures.append("summary.json scientific_claim_allowed is false")
    return {
        "schema": "neurotwin.prepared_evidence_gate.v1",
        "stage": stage,
        "passed": False if failures else summary_claim,
        "scientific_claim_allowed": summary_claim,
        "summary_is_source_of_truth": True,
        "failures": failures,
        "checks": {
            "quarantined_tasks": quarantined or [],
            "eval_audit": eval_audit or {},
            "stimulus_evidence": stimulus or {},
            "baseline_ranking_present": baseline_ranking_present,
            "eval_audit_present": bool(eval_audit),
            "eval_audit_passed": bool(eval_audit.get("passed")) if isinstance(eval_audit, dict) else False,
            "competitor_reproduction_status_present": competitor_reproduction_status_present,
            "paper_mode_gate_present": paper_mode_gate_present,
        },
    }


def read_json_artifact(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"error": "invalid_json", "path": str(path), "message": str(exc)}
    except OSError as exc:
        return {"error": "read_failed", "path": str(path), "message": str(exc)}


def _baseline_ranking_present(run_dir: Path | None) -> bool:
    if run_dir is None:
        return False
    if _prepared_suite_has_rankings(read_json_artifact(run_dir / "prepared_baseline_suite.json")):
        return True
    return _baseline_csv_has_rankings(run_dir / "tables" / "baseline_ranking.csv")


def _prepared_suite_has_rankings(payload: Any) -> bool:
    tasks = payload.get("tasks", {}) if isinstance(payload, dict) else {}
    if not isinstance(tasks, dict):
        return False
    for task_payload in tasks.values():
        ranking = task_payload.get("ranking") if isinstance(task_payload, dict) else None
        if isinstance(ranking, list) and any(_real_ranking_row(row) for row in ranking):
            return True
    return False


def _baseline_csv_has_rankings(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (csv.Error, OSError):
        return False
    required = {"task_id", "model_id", "metric", "value", "rank"}
    if not rows or not required.issubset(rows[0].keys()):
        return False
    return any(_real_ranking_row(row) for row in rows)


def _real_ranking_row(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    if str(row.get("task_id", "")).strip() == "baseline_ranking_unavailable":
        return False
    return bool(str(row.get("model_id", "")).strip() and str(row.get("metric", "")).strip() and str(row.get("rank", "")).strip())


def _competitor_reproduction_status_present(run_dir: Path | None) -> bool:
    if run_dir is None:
        return False
    prepared_suite = read_json_artifact(run_dir / "prepared_baseline_suite.json")
    catalog = prepared_suite.get("baseline_catalog") if isinstance(prepared_suite, dict) else None
    return isinstance(catalog, list) and bool(catalog)


def _paper_mode_gate_present(run_dir: Path | None) -> bool:
    if run_dir is None:
        return False
    gate = read_json_artifact(run_dir / "paper_mode_gate.json")
    return isinstance(gate, dict) and gate.get("passed") is True


def format_evidence_diagnostic_report(summary: dict[str, Any], evidence_gate: dict[str, Any]) -> str:
    lines = [
        "# NeuroTwin Prepared Run Diagnostic Report",
        "",
        f"- status: {summary.get('status', 'unknown')}",
        f"- task_id: {summary.get('task_id', 'unknown')}",
        f"- source_modality: {summary.get('source_modality', 'unknown')}",
        f"- target_modality: {summary.get('target_modality', 'unknown')}",
        f"- scientific_claim_allowed: {summary.get('scientific_claim_allowed', False)}",
        f"- evidence_gate_passed: {evidence_gate.get('passed', False)}",
        "",
        "## Gate Failures",
        "",
    ]
    failures = evidence_gate.get("failures")
    if isinstance(failures, list) and failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- none")
    lines.extend(["", "## Stimulus Evidence", ""])
    stimulus = summary.get("stimulus_evidence")
    if isinstance(stimulus, dict) and stimulus:
        for key in (
            "status",
            "claim_eligible",
            "require_real_stimulus",
            "source_artifact_hash_verified",
            "hash_verified",
            "claim_note",
        ):
            lines.append(f"- {key}: {stimulus.get(key, 'unknown')}")
    else:
        lines.append("- missing")
    lines.extend(["", "## Quarantined Tasks", ""])
    quarantined = summary.get("quarantined_tasks")
    if isinstance(quarantined, (list, tuple)) and quarantined:
        for row in quarantined:
            if isinstance(row, dict):
                lines.append(f"- {row.get('task_id', 'unknown')}: {row.get('reason', 'unknown')}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _task_result_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("task_results")
    if not isinstance(rows, (list, tuple)):
        return []
    return [row for row in rows if isinstance(row, dict)]
