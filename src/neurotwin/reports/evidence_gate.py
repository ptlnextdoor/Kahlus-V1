from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neurotwin.eval.claim_contracts import TaskClaimContract, task_claim_contract
from neurotwin.repro import write_json
from neurotwin.eval.forecast_eligibility import (
    ForecastEligibilityDecision,
    validate_forecast_eligibility_artifact,
)
from neurotwin.eval.paper_gate import paper_mode_gate_allows_claim_for_run


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
    (path / "diagnostic_report.md").write_text(
        format_evidence_diagnostic_report(summary, evidence_gate), encoding="utf-8"
    )
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
        failed_tasks = ",".join(
            str(row.get("task_id", "unknown"))
            for row in quarantined
            if isinstance(row, dict)
        )
        failures.append(f"required task quarantined: {failed_tasks or 'unknown'}")
    task_rows = _task_result_rows(summary)
    task_contracts: list[TaskClaimContract] = []
    if not task_rows:
        failures.append("summary has no claim-bearing task results")
    for row in task_rows:
        task_id = str(row.get("task_id", "")).strip()
        contract = task_claim_contract(task_id)
        if contract is None:
            failures.append(f"unknown task claim contract: {task_id or 'missing'}")
            continue
        task_contracts.append(contract)
        missing_metrics = [
            field
            for field in contract.required_metric_fields
            if not _finite_number(row.get(field))
        ]
        if missing_metrics:
            failures.append(
                f"required task {task_id} is missing finite metric(s): {','.join(missing_metrics)}"
            )
    stimulus = summary.get("stimulus_evidence")
    if (
        isinstance(stimulus, dict)
        and stimulus
        and not bool(stimulus.get("claim_eligible"))
    ):
        failures.append("stimulus-to-fMRI evidence is not claim eligible")
    baseline_suite = (
        read_json_artifact(run_path / "prepared_baseline_suite.json")
        if run_path is not None
        else {}
    )
    baseline_failures = _baseline_contract_failures(baseline_suite, task_contracts)
    failures.extend(baseline_failures)
    baseline_ranking_present = bool(task_contracts) and not baseline_failures
    competitor_reproduction_status_present = _required_catalog_entries_present(
        baseline_suite, task_contracts
    )
    paper_mode_gate_present = _paper_mode_gate_present(run_path)
    forecast_required = any(
        contract.requires_forecast_eligibility for contract in task_contracts
    )
    forecast_eligibility = (
        _forecast_eligibility(run_path) if forecast_required else None
    )
    forecast_eligibility_passed = bool(
        forecast_eligibility and forecast_eligibility.claim_eligible
    )
    if not task_contracts:
        baseline_ranking_present = False
    if not competitor_reproduction_status_present:
        failures.append(
            "required task-specific baseline catalog entries are missing or unavailable"
        )
    if not paper_mode_gate_present:
        failures.append("paper_mode_gate.json missing or not passed")
    if forecast_required and not forecast_eligibility_passed:
        details = (
            "; ".join(forecast_eligibility.violations)
            if forecast_eligibility is not None
            else "artifact missing"
        )
        failures.append(f"forecast eligibility missing or failed: {details}")
    summary_claim = bool(summary.get("scientific_claim_allowed"))
    if not summary_claim:
        failures.append("summary.json scientific_claim_allowed is false")
    return {
        "schema": "neurotwin.prepared_evidence_gate.v1",
        "stage": stage,
        "passed": False if failures else summary_claim,
        "scientific_claim_allowed": summary_claim,
        "summary_is_source_of_truth": False,
        "failures": failures,
        "checks": {
            "quarantined_tasks": quarantined or [],
            "eval_audit": eval_audit or {},
            "stimulus_evidence": stimulus or {},
            "baseline_ranking_present": baseline_ranking_present,
            "eval_audit_present": bool(eval_audit),
            "eval_audit_passed": bool(eval_audit.get("passed"))
            if isinstance(eval_audit, dict)
            else False,
            "competitor_reproduction_status_present": competitor_reproduction_status_present,
            "paper_mode_gate_present": paper_mode_gate_present,
            "forecast_eligibility_required": forecast_required,
            "forecast_eligibility_passed": forecast_eligibility_passed,
            "forecast_eligibility_violations": list(forecast_eligibility.violations)
            if forecast_eligibility
            else [],
            "task_contracts": [contract.to_dict() for contract in task_contracts],
            "baseline_contract_failures": baseline_failures,
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


def _baseline_contract_failures(
    payload: Any, contracts: list[TaskClaimContract]
) -> list[str]:
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(tasks, dict):
        return ["prepared baseline suite has no task rankings"] if contracts else []
    failures: list[str] = []
    for contract in contracts:
        task_payload = tasks.get(contract.task_id)
        ranking = (
            task_payload.get("ranking") if isinstance(task_payload, dict) else None
        )
        if not isinstance(ranking, list) or not ranking:
            failures.append(f"baseline ranking missing for task {contract.task_id}")
            continue
        ranked_models = {
            str(row.get("model_id"))
            for row in ranking
            if isinstance(row, dict)
            and str(row.get("model_id", "")).strip()
            and str(row.get("metric", "")).strip()
            and isinstance(row.get("rank"), int)
        }
        required = set(contract.required_baselines) | set(contract.required_controls)
        missing = sorted(required - ranked_models)
        if missing:
            failures.append(
                f"task {contract.task_id} missing required ranked baseline/control(s): {','.join(missing)}"
            )
    return failures


def _required_catalog_entries_present(
    payload: Any, contracts: list[TaskClaimContract]
) -> bool:
    if not contracts or not isinstance(payload, dict):
        return False
    catalog = payload.get("baseline_catalog")
    if not isinstance(catalog, list):
        return False
    available = {
        str(row.get("model_id"))
        for row in catalog
        if isinstance(row, dict)
        and str(row.get("model_id", "")).strip()
        and row.get("status") not in {None, "", "unavailable"}
    }
    required = {
        model_id
        for contract in contracts
        for model_id in (*contract.required_baselines, *contract.required_controls)
    }
    return required <= available


def _paper_mode_gate_present(run_dir: Path | None) -> bool:
    if run_dir is None:
        return False
    return paper_mode_gate_allows_claim_for_run(run_dir)


def _forecast_eligibility(run_dir: Path | None) -> ForecastEligibilityDecision | None:
    if run_dir is None:
        return None
    payload = read_json_artifact(run_dir / "forecast_eligibility.json")
    return validate_forecast_eligibility_artifact(
        payload if isinstance(payload, dict) else None
    )


def format_evidence_diagnostic_report(
    summary: dict[str, Any], evidence_gate: dict[str, Any]
) -> str:
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
                lines.append(
                    f"- {row.get('task_id', 'unknown')}: {row.get('reason', 'unknown')}"
                )
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _task_result_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("task_results")
    if not isinstance(rows, (list, tuple)):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value == value and value not in {float("inf"), float("-inf")}
