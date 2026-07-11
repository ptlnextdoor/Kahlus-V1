from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from neurotwin.contracts.paper_mode import CANONICAL_REQUIRED_SEEDS
from neurotwin.eval.paper_contracts import (
    PaperModeEvidence,
    normalize_seed_tuple,
    validate_paper_mode_evidence,
)
from neurotwin.eval.forecast_eligibility import validate_forecast_eligibility_artifact


@dataclass(frozen=True)
class PaperModeGateReport:
    passed: bool
    violations: tuple[str, ...]
    warnings: tuple[str, ...]
    checked: tuple[str, ...]
    required_seeds: tuple[int, ...]
    observed_seeds: tuple[int, ...]
    require_ci: bool
    forecast_eligibility_required: bool
    forecast_eligibility_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PaperModeGateError(ValueError):
    def __init__(self, report: PaperModeGateReport):
        super().__init__("paper-mode gate failed: " + "; ".join(report.violations))
        self.report = report


def validate_paper_mode_payload(
    payload: PaperModeEvidence | dict[str, Any],
    audit_report: Any | None = None,
    require_ci: bool = True,
    raise_on_fail: bool = False,
) -> PaperModeGateReport:
    """Validate the artifact contract required before paper-mode claims."""

    gate = validate_paper_mode_evidence(payload, audit_report=audit_report, require_ci=require_ci)
    forecast_required = _contains_forecast_task(payload)
    forecast_decision = validate_forecast_eligibility_artifact(
        payload.get("forecast_eligibility") if isinstance(payload, dict) else None
    )
    violations = list(gate.violations)
    checked = list(gate.checked)
    if forecast_required:
        checked.append("forecast_eligibility")
        violations.extend(forecast_decision.violations)
    report = PaperModeGateReport(
        passed=not violations,
        violations=tuple(violations),
        warnings=gate.warnings,
        checked=tuple(checked),
        required_seeds=gate.required_seeds,
        observed_seeds=gate.observed_seeds,
        require_ci=gate.require_ci,
        forecast_eligibility_required=forecast_required,
        forecast_eligibility_passed=forecast_decision.claim_eligible if forecast_required else True,
    )
    if raise_on_fail and not report.passed:
        raise PaperModeGateError(report)
    return report


def format_paper_mode_gate(report: PaperModeGateReport) -> str:
    lines = [
        "paper_mode_gate=True",
        f"paper_mode_passed={report.passed}",
        "required_seeds=" + ",".join(str(seed) for seed in report.required_seeds),
        "observed_seeds=" + ",".join(str(seed) for seed in report.observed_seeds),
        f"require_ci={report.require_ci}",
        f"forecast_eligibility_required={report.forecast_eligibility_required}",
        f"forecast_eligibility_passed={report.forecast_eligibility_passed}",
        "checked=" + ",".join(report.checked),
    ]
    for violation in report.violations:
        lines.append(f"paper_mode_violation={violation}")
    for warning in report.warnings:
        lines.append(f"paper_mode_warning={warning}")
    return "\n".join(lines)


def paper_mode_gate_allows_claim(payload: PaperModeGateReport | dict[str, Any] | None) -> bool:
    if isinstance(payload, PaperModeGateReport):
        required_seeds = payload.required_seeds
        observed_seeds = payload.observed_seeds
        violations = payload.violations
        require_ci = payload.require_ci
        passed = payload.passed
        forecast_required = payload.forecast_eligibility_required
        forecast_passed = payload.forecast_eligibility_passed
    elif isinstance(payload, dict):
        required_seeds = normalize_seed_tuple(payload.get("required_seeds"))
        observed_seeds = normalize_seed_tuple(payload.get("observed_seeds"))
        violations = payload.get("violations")
        require_ci = payload.get("require_ci")
        passed = payload.get("passed")
        forecast_required = payload.get("forecast_eligibility_required")
        forecast_passed = payload.get("forecast_eligibility_passed")
    else:
        return False
    if passed is not True or require_ci is not True:
        return False
    if forecast_required not in (True, False):
        return False
    if forecast_required is True and forecast_passed is not True:
        return False
    if not isinstance(violations, (list, tuple)) or any(str(item).strip() for item in violations):
        return False
    if required_seeds != CANONICAL_REQUIRED_SEEDS or observed_seeds is None:
        return False
    return all(seed in observed_seeds for seed in CANONICAL_REQUIRED_SEEDS)


def effective_scientific_claim_allowed(
    summary: dict[str, Any] | None,
    gate_payload: PaperModeGateReport | dict[str, Any] | None,
) -> bool:
    if not isinstance(summary, dict):
        return False
    if summary.get("scientific_claim_allowed") is not True:
        return False
    if summary.get("synthetic_only") is not False:
        return False
    if summary.get("real_data_smoke") is not False:
        return False
    return paper_mode_gate_allows_claim(gate_payload)


def load_run_summary(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "summary.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _json_artifact_error(path, "invalid_json", str(exc))
    except OSError as exc:
        return _json_artifact_error(path, "read_failed", str(exc))
    return payload if isinstance(payload, dict) else {}


def load_paper_mode_gate(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "paper_mode_gate.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _json_artifact_error(path, "invalid_json", str(exc))
    except OSError as exc:
        return _json_artifact_error(path, "read_failed", str(exc))
    return payload if isinstance(payload, dict) else {}


def effective_scientific_claim_allowed_for_run(
    run_dir: str | Path,
    summary: dict[str, Any] | None = None,
) -> bool:
    summary_payload = summary if isinstance(summary, dict) else load_run_summary(run_dir)
    return effective_scientific_claim_allowed(summary_payload, load_paper_mode_gate(run_dir))


def _json_artifact_error(path: Path, error: str, message: str) -> dict[str, Any]:
    return {"error": error, "path": str(path), "message": message}


def _contains_forecast_task(payload: Any) -> bool:
    if isinstance(payload, PaperModeEvidence):
        return _contains_forecast_task(payload.seed_results)
    if isinstance(payload, dict):
        task_id = payload.get("task_id")
        if isinstance(task_id, str) and "forecast" in task_id.lower():
            return True
        tasks = payload.get("tasks")
        if isinstance(tasks, dict) and any("forecast" in str(key).lower() for key in tasks):
            return True
        return any(_contains_forecast_task(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return any(_contains_forecast_task(value) for value in payload)
    return False
