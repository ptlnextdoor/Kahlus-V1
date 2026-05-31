from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


CANONICAL_REQUIRED_SEEDS = (0, 1, 2)
CONCRETE_SEED_CONTAINER_KEYS = ("runs", "seed_results", "per_seed_results", "baseline_runs", "per_seed_baselines")
RESULT_EVIDENCE_KEYS = ("aggregate", "tasks", "task_results", "metrics_by_model", "metrics", "ranking", "baseline_suite")
REPORT_METRIC_TOKENS = ("mse", "mae", "pearsonr", "spearmanr", "r2", "error", "gain", "accuracy", "recall")


@dataclass(frozen=True)
class PaperModeGateReport:
    passed: bool
    violations: tuple[str, ...]
    warnings: tuple[str, ...]
    checked: tuple[str, ...]
    required_seeds: tuple[int, ...]
    observed_seeds: tuple[int, ...]
    require_ci: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PaperModeGateError(ValueError):
    def __init__(self, report: PaperModeGateReport):
        super().__init__("paper-mode gate failed: " + "; ".join(report.violations))
        self.report = report


def validate_paper_mode_payload(
    payload: dict[str, Any],
    audit_report: Any | None = None,
    require_ci: bool = True,
    raise_on_fail: bool = False,
) -> PaperModeGateReport:
    """Validate the artifact contract required before paper-mode claims.

    This is intentionally a contract gate, not the three-seed aggregation
    implementation. Task 3 can satisfy it by writing all required seed results
    and CI summaries into the same payload shape.
    """

    checked = ("eval_audit", "baseline_rankings", "required_seeds", "ci_summaries")
    violations: list[str] = []
    warnings: list[str] = []

    audit_payload = _audit_payload(audit_report, payload)
    if audit_payload is None:
        violations.append("paper mode requires a prepared eval audit payload")
    elif not bool(audit_payload.get("passed")):
        details = audit_payload.get("violations")
        suffix = f": {details}" if details else ""
        violations.append(f"prepared eval audit did not pass{suffix}")

    if not _aggregate_rank(payload):
        violations.append("baseline aggregate_rank is empty")

    observed_seeds = _observed_seeds(payload)
    missing = tuple(seed for seed in CANONICAL_REQUIRED_SEEDS if seed not in observed_seeds)
    if missing:
        violations.append(
            "paper mode requires seeds "
            + ",".join(str(seed) for seed in CANONICAL_REQUIRED_SEEDS)
            + "; missing "
            + ",".join(str(seed) for seed in missing)
        )

    if require_ci:
        violations.extend(_ci_violations(payload))

    report = PaperModeGateReport(
        passed=not violations,
        violations=tuple(violations),
        warnings=tuple(warnings),
        checked=checked,
        required_seeds=CANONICAL_REQUIRED_SEEDS,
        observed_seeds=observed_seeds,
        require_ci=bool(require_ci),
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
        "checked=" + ",".join(report.checked),
    ]
    for violation in report.violations:
        lines.append(f"paper_mode_violation={violation}")
    for warning in report.warnings:
        lines.append(f"paper_mode_warning={warning}")
    return "\n".join(lines)


def _audit_payload(audit_report: Any | None, payload: dict[str, Any]) -> dict[str, Any] | None:
    if audit_report is not None:
        if hasattr(audit_report, "to_dict"):
            audit_report = audit_report.to_dict()
        elif hasattr(audit_report, "passed"):
            audit_report = {
                "passed": bool(getattr(audit_report, "passed")),
                "violations": tuple(getattr(audit_report, "violations", ())),
            }
        return audit_report if isinstance(audit_report, dict) else None
    for key in ("eval_audit", "prepared_eval_audit", "audit"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return None


def _aggregate_rank(payload: dict[str, Any]) -> list[Any]:
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict):
        return []
    ranking = aggregate.get("aggregate_rank")
    return ranking if isinstance(ranking, list) else []


def _observed_seeds(payload: dict[str, Any]) -> tuple[int, ...]:
    seeds: set[int] = set()
    _add_concrete_seed_record(seeds, payload)
    for record in _iter_concrete_seed_records(payload):
        _add_concrete_seed_record(seeds, record)
    return tuple(sorted(seeds))


def _iter_concrete_seed_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key in CONCRETE_SEED_CONTAINER_KEYS:
        records.extend(_records_from_container(payload.get(key)))
    tasks = payload.get("tasks")
    if isinstance(tasks, dict):
        for task_result in tasks.values():
            if isinstance(task_result, dict):
                for key in CONCRETE_SEED_CONTAINER_KEYS:
                    records.extend(_records_from_container(task_result.get(key)))
    aggregate = payload.get("aggregate")
    if isinstance(aggregate, dict):
        for key in CONCRETE_SEED_CONTAINER_KEYS:
            records.extend(_records_from_container(aggregate.get(key)))
    return records


def _records_from_container(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        if "seed" in value:
            return [value]
        return [item for item in value.values() if isinstance(item, dict)]
    return []


def _add_concrete_seed_record(seeds: set[int], record: dict[str, Any]) -> None:
    if not _has_result_evidence(record):
        return
    seed = _coerce_seed(record.get("seed"))
    if seed is not None:
        seeds.add(seed)


def _coerce_seed(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _has_result_evidence(record: dict[str, Any]) -> bool:
    for key in RESULT_EVIDENCE_KEYS:
        value = record.get(key)
        if isinstance(value, dict) and value:
            if key == "aggregate":
                ranking = value.get("aggregate_rank")
                if isinstance(ranking, list) and ranking:
                    return True
                continue
            return True
        if isinstance(value, list) and value:
            return True
    return False


def _ci_violations(payload: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    violations.extend(_test_ci_violations(payload, "top-level test reporting"))
    tasks = payload.get("tasks")
    if isinstance(tasks, dict):
        for task_id, result in sorted(tasks.items()):
            if not isinstance(result, dict):
                continue
            metrics = result.get("metrics")
            if isinstance(metrics, dict):
                violations.extend(_report_metric_ci_violations(metrics, f"{task_id}:metrics"))
            ranking = result.get("ranking")
            if not isinstance(ranking, list) or not ranking:
                continue
            metrics_by_model = result.get("metrics_by_model")
            if not isinstance(metrics_by_model, dict):
                violations.append(f"{task_id} lacks metrics_by_model for ranked baselines")
                continue
            for row in ranking:
                if not isinstance(row, dict):
                    continue
                model_id = str(row.get("model_id"))
                metrics = metrics_by_model.get(model_id)
                if not isinstance(metrics, dict) or not _has_finite_ci(metrics, "mse"):
                    violations.append(f"{task_id}:{model_id} lacks finite mse CI summary")
    task_results = payload.get("task_results")
    if isinstance(task_results, list):
        for row in task_results:
            if isinstance(row, dict):
                violations.extend(_test_ci_violations(row, str(row.get("task_id", "task_result"))))
    return violations


def _has_finite_ci(metrics: dict[str, Any], metric: str) -> bool:
    low = metrics.get(f"{metric}_ci_low")
    high = metrics.get(f"{metric}_ci_high")
    return _finite_number(low) and _finite_number(high)


def _test_ci_violations(row: dict[str, Any], label: str) -> list[str]:
    violations: list[str] = []
    for key, value in sorted(row.items()):
        if not _is_test_metric_key(key, value):
            continue
        if not _has_finite_ci(row, key):
            violations.append(f"{label}:{key} lacks finite CI summary")
    return violations


def _report_metric_ci_violations(metrics: dict[str, Any], label: str) -> list[str]:
    violations: list[str] = []
    for key, value in sorted(metrics.items()):
        if not _is_report_metric_key(key, value):
            continue
        if not _has_finite_ci(metrics, key):
            violations.append(f"{label}:{key} lacks finite CI summary")
    return violations


def _is_test_metric_key(key: str, value: Any) -> bool:
    return (
        key.startswith("test_")
        and not key.endswith(("_ci_low", "_ci_high"))
        and isinstance(value, (int, float, np.floating))
        and not isinstance(value, bool)
    )


def _is_report_metric_key(key: str, value: Any) -> bool:
    return (
        not key.endswith(("_ci_low", "_ci_high"))
        and isinstance(value, (int, float, np.floating))
        and not isinstance(value, bool)
        and any(token in key for token in REPORT_METRIC_TOKENS)
    )


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.floating)) and np.isfinite(float(value))
