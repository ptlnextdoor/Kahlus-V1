from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

from neurotwin.contracts.paper_mode import CANONICAL_REQUIRED_SEEDS
from neurotwin.eval.paper_contracts import (
    PaperModeEvidence,
    normalize_seed_tuple,
    validate_paper_mode_evidence,
)
from neurotwin.eval.claim_contracts import (
    claim_contract_sha256,
    collect_task_claim_contracts,
)
from neurotwin.eval.forecast_eligibility import (
    forecast_eligibility_sha256,
    validate_forecast_eligibility_artifact,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PaperModeGateReport:
    passed: bool
    violations: tuple[str, ...]
    warnings: tuple[str, ...]
    checked: tuple[str, ...]
    required_seeds: tuple[int, ...]
    observed_seeds: tuple[int, ...]
    require_ci: bool
    claim_contract_sha256: str
    forecast_eligibility_required: bool
    forecast_eligibility_passed: bool
    forecast_eligibility_sha256: str | None

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

    gate = validate_paper_mode_evidence(
        payload, audit_report=audit_report, require_ci=require_ci
    )
    task_contracts, unknown_task_ids = collect_task_claim_contracts(payload)
    forecast_required = any(
        contract.requires_forecast_eligibility for contract in task_contracts
    )
    eligibility_payload = (
        payload.get("forecast_eligibility") if isinstance(payload, dict) else None
    )
    forecast_decision = validate_forecast_eligibility_artifact(eligibility_payload)
    violations = list(gate.violations)
    checked = list(gate.checked)
    checked.append("task_claim_contract")
    if unknown_task_ids:
        violations.append(
            "unknown task claim contract(s): " + ",".join(unknown_task_ids)
        )
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
        claim_contract_sha256=claim_contract_sha256(task_contracts),
        forecast_eligibility_required=forecast_required,
        forecast_eligibility_passed=forecast_decision.claim_eligible
        if forecast_required
        else True,
        forecast_eligibility_sha256=forecast_eligibility_sha256(eligibility_payload)
        if forecast_required
        else None,
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
        f"claim_contract_sha256={report.claim_contract_sha256}",
        f"forecast_eligibility_required={report.forecast_eligibility_required}",
        f"forecast_eligibility_passed={report.forecast_eligibility_passed}",
        f"forecast_eligibility_sha256={report.forecast_eligibility_sha256 or 'not_required'}",
        "checked=" + ",".join(report.checked),
    ]
    for violation in report.violations:
        lines.append(f"paper_mode_violation={violation}")
    for warning in report.warnings:
        lines.append(f"paper_mode_warning={warning}")
    return "\n".join(lines)


def paper_mode_gate_allows_claim(
    payload: PaperModeGateReport | dict[str, Any] | None,
    *,
    task_payload: Any | None,
    forecast_eligibility: dict[str, Any] | None = None,
) -> bool:
    if isinstance(payload, PaperModeGateReport):
        required_seeds = payload.required_seeds
        observed_seeds = payload.observed_seeds
        violations = payload.violations
        require_ci = payload.require_ci
        passed = payload.passed
        contract_hash = payload.claim_contract_sha256
        forecast_required = payload.forecast_eligibility_required
        forecast_passed = payload.forecast_eligibility_passed
        eligibility_hash = payload.forecast_eligibility_sha256
    elif isinstance(payload, dict):
        required_seeds = normalize_seed_tuple(payload.get("required_seeds"))
        observed_seeds = normalize_seed_tuple(payload.get("observed_seeds"))
        violations = payload.get("violations")
        require_ci = payload.get("require_ci")
        passed = payload.get("passed")
        contract_hash = payload.get("claim_contract_sha256")
        forecast_required = payload.get("forecast_eligibility_required")
        forecast_passed = payload.get("forecast_eligibility_passed")
        eligibility_hash = payload.get("forecast_eligibility_sha256")
    else:
        return False
    if passed is not True or require_ci is not True:
        return False
    task_contracts, unknown_task_ids = collect_task_claim_contracts(task_payload)
    if unknown_task_ids or not task_contracts:
        return False
    expected_contract_hash = claim_contract_sha256(task_contracts)
    if (
        not isinstance(contract_hash, str)
        or not _SHA256_RE.fullmatch(contract_hash)
        or contract_hash != expected_contract_hash
    ):
        return False
    expected_forecast_required = any(
        contract.requires_forecast_eligibility for contract in task_contracts
    )
    if (
        forecast_required not in (True, False)
        or forecast_required is not expected_forecast_required
    ):
        return False
    if forecast_required is True:
        observed_eligibility_hash = forecast_eligibility_sha256(forecast_eligibility)
        if (
            forecast_passed is not True
            or observed_eligibility_hash is None
            or eligibility_hash != observed_eligibility_hash
        ):
            return False
    elif eligibility_hash is not None:
        return False
    if not isinstance(violations, (list, tuple)) or any(
        str(item).strip() for item in violations
    ):
        return False
    if required_seeds != CANONICAL_REQUIRED_SEEDS or observed_seeds is None:
        return False
    return all(seed in observed_seeds for seed in CANONICAL_REQUIRED_SEEDS)


def effective_scientific_claim_allowed(
    summary: dict[str, Any] | None,
    gate_payload: PaperModeGateReport | dict[str, Any] | None,
    *,
    task_payload: Any | None,
    forecast_eligibility: dict[str, Any] | None = None,
) -> bool:
    if not isinstance(summary, dict):
        return False
    if summary.get("scientific_claim_allowed") is not True:
        return False
    if summary.get("synthetic_only") is not False:
        return False
    if summary.get("real_data_smoke") is not False:
        return False
    return paper_mode_gate_allows_claim(
        gate_payload,
        task_payload=task_payload,
        forecast_eligibility=forecast_eligibility,
    )


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


def load_prepared_baseline_suite(run_dir: str | Path) -> dict[str, Any]:
    return _load_json_object(Path(run_dir) / "prepared_baseline_suite.json")


def load_forecast_eligibility(run_dir: str | Path) -> dict[str, Any]:
    return _load_json_object(Path(run_dir) / "forecast_eligibility.json")


def paper_mode_gate_allows_claim_for_run(run_dir: str | Path) -> bool:
    return paper_mode_gate_allows_claim(
        load_paper_mode_gate(run_dir),
        task_payload=load_prepared_baseline_suite(run_dir),
        forecast_eligibility=load_forecast_eligibility(run_dir),
    )


def effective_scientific_claim_allowed_for_run(
    run_dir: str | Path,
    summary: dict[str, Any] | None = None,
) -> bool:
    summary_payload = (
        summary if isinstance(summary, dict) else load_run_summary(run_dir)
    )
    return effective_scientific_claim_allowed(
        summary_payload,
        load_paper_mode_gate(run_dir),
        task_payload=load_prepared_baseline_suite(run_dir),
        forecast_eligibility=load_forecast_eligibility(run_dir),
    )


def _json_artifact_error(path: Path, error: str, message: str) -> dict[str, Any]:
    return {"error": error, "path": str(path), "message": message}


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _json_artifact_error(path, "invalid_json", str(exc))
    except OSError as exc:
        return _json_artifact_error(path, "read_failed", str(exc))
    return payload if isinstance(payload, dict) else {}
