from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


CANONICAL_REQUIRED_SEEDS = (0, 1, 2)
CONCRETE_SEED_CONTAINER_KEYS = ("runs", "seed_results", "per_seed_results", "baseline_runs", "per_seed_baselines")


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

    metric_evidence_check = "ci_summaries" if require_ci else "metric_evidence_without_ci"
    checked = ("eval_audit", "baseline_rankings", "required_seeds", metric_evidence_check)
    violations: list[str] = []
    warnings: list[str] = []

    audit_payload = _audit_payload(audit_report, payload)
    if audit_payload is None:
        violations.append("paper mode requires a prepared eval audit payload")
    elif not bool(audit_payload.get("passed")):
        details = audit_payload.get("violations")
        suffix = f": {details}" if details else ""
        violations.append(f"prepared eval audit did not pass{suffix}")

    aggregate_rank = _aggregate_rank(payload)
    seed_records = _iter_concrete_seed_records(payload)
    derived_rank = _aggregate_rank_from_seed_records(seed_records)

    if not aggregate_rank:
        violations.append("baseline aggregate_rank is empty")
    elif not derived_rank:
        violations.append("paper mode requires per-seed ranked baseline evidence")
    elif not _aggregate_rank_matches(aggregate_rank, derived_rank):
        violations.append("baseline aggregate_rank does not match per-seed ranked baseline evidence")

    observed_seeds, seed_evidence_violations = _observed_seeds(seed_records, require_ci=bool(require_ci))
    missing = tuple(seed for seed in CANONICAL_REQUIRED_SEEDS if seed not in observed_seeds)
    if missing:
        violations.append(
            "paper mode requires seeds "
            + ",".join(str(seed) for seed in CANONICAL_REQUIRED_SEEDS)
            + "; missing "
            + ",".join(str(seed) for seed in missing)
        )
    for seed in missing:
        violations.extend(seed_evidence_violations.get(seed, ()))

    if require_ci:
        violations.extend(_seed_aggregate_ci_violations(payload, aggregate_rank))

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


def paper_mode_gate_allows_claim(payload: PaperModeGateReport | dict[str, Any] | None) -> bool:
    if isinstance(payload, PaperModeGateReport):
        required_seeds = payload.required_seeds
        observed_seeds = payload.observed_seeds
        violations = payload.violations
        require_ci = payload.require_ci
        passed = payload.passed
    elif isinstance(payload, dict):
        required_seeds = _normalize_seed_tuple(payload.get("required_seeds"))
        observed_seeds = _normalize_seed_tuple(payload.get("observed_seeds"))
        violations = payload.get("violations")
        require_ci = payload.get("require_ci")
        passed = payload.get("passed")
    else:
        return False
    if passed is not True or require_ci is not True:
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
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_paper_mode_gate(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "paper_mode_gate.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def effective_scientific_claim_allowed_for_run(
    run_dir: str | Path,
    summary: dict[str, Any] | None = None,
) -> bool:
    summary_payload = summary if isinstance(summary, dict) else load_run_summary(run_dir)
    return effective_scientific_claim_allowed(summary_payload, load_paper_mode_gate(run_dir))


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


def _observed_seeds(
    seed_records: list[dict[str, Any]],
    require_ci: bool = True,
) -> tuple[tuple[int, ...], dict[int, tuple[str, ...]]]:
    observed: set[int] = set()
    violations: dict[int, list[str]] = {}
    for record in seed_records:
        seed = _coerce_seed(record.get("seed"))
        if seed not in CANONICAL_REQUIRED_SEEDS:
            continue
        passed, record_violations = _seed_record_has_ranked_baseline_evidence(record, require_ci=require_ci)
        if passed:
            observed.add(seed)
        elif seed is not None:
            violations.setdefault(seed, []).extend(record_violations)
    return tuple(sorted(observed)), {seed: tuple(values) for seed, values in violations.items()}


def _iter_concrete_seed_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key in CONCRETE_SEED_CONTAINER_KEYS:
        records.extend(_records_from_container(payload.get(key)))
    return records


def _records_from_container(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        if "seed" in value:
            return [value]
        records = []
        for key, item in value.items():
            if not isinstance(item, dict):
                continue
            if "seed" in item:
                records.append(item)
                continue
            seed = _coerce_seed(key)
            if seed is None:
                records.append(item)
            else:
                records.append({"seed": seed, **item})
        return records
    return []


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


def _normalize_seed_tuple(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    normalized: list[int] = []
    for item in value:
        seed = _coerce_seed(item)
        if seed is None:
            return None
        normalized.append(seed)
    return tuple(normalized)


def _seed_record_has_ranked_baseline_evidence(
    record: dict[str, Any],
    require_ci: bool = True,
) -> tuple[bool, tuple[str, ...]]:
    seed = record.get("seed", "unknown")
    tasks = record.get("tasks")
    if not isinstance(tasks, dict):
        return False, (f"seed {seed} lacks task payloads with ranked baselines",)

    saw_ranked_task = False
    violations: list[str] = []
    for task_id, task_result in sorted(tasks.items()):
        if not isinstance(task_result, dict):
            continue
        ranking = task_result.get("ranking")
        metrics_by_model = task_result.get("metrics_by_model")
        if not isinstance(ranking, list) or not ranking:
            continue
        saw_ranked_task = True
        if not isinstance(metrics_by_model, dict):
            violations.append(f"seed {seed}:{task_id} lacks metrics_by_model for ranked baselines")
            continue
        for row in ranking:
            if not isinstance(row, dict):
                continue
            model_id = row.get("model_id")
            metric = str(row.get("metric") or "mse")
            metrics = metrics_by_model.get(model_id) if model_id is not None else None
            if not isinstance(metrics, dict):
                violations.append(f"seed {seed}:{task_id}:{model_id} lacks metrics for ranked baseline")
                continue
            if not _finite_number(metrics.get(metric)):
                violations.append(f"seed {seed}:{task_id}:{model_id} lacks finite {metric} metric")
            if require_ci and not _has_finite_ci(metrics, metric):
                violations.append(f"seed {seed}:{task_id}:{model_id} lacks finite {metric} CI summary")
    if not saw_ranked_task:
        violations.append(f"seed {seed} lacks ranked baseline evidence")
    return saw_ranked_task and not violations, tuple(violations)


def _aggregate_rank_from_seed_records(seed_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranks_by_model: dict[str, list[float]] = {}
    for record in seed_records:
        tasks = record.get("tasks")
        if not isinstance(tasks, dict):
            continue
        for task_result in tasks.values():
            if not isinstance(task_result, dict):
                continue
            ranking = task_result.get("ranking")
            metrics_by_model = task_result.get("metrics_by_model")
            if not isinstance(ranking, list) or not isinstance(metrics_by_model, dict):
                continue
            for row in ranking:
                if not isinstance(row, dict):
                    continue
                model_id = row.get("model_id")
                metric = str(row.get("metric") or "mse")
                if model_id is None:
                    continue
                metrics = metrics_by_model.get(model_id)
                if not isinstance(metrics, dict) or not _finite_number(metrics.get(metric)):
                    continue
                rank = _finite_rank(row)
                if rank is not None:
                    ranks_by_model.setdefault(str(model_id), []).append(rank)
    return sorted(
        (
            {
                "model_id": model_id,
                "mean_rank": float(np.mean(ranks)),
                "tasks_ranked": len(ranks),
            }
            for model_id, ranks in ranks_by_model.items()
            if ranks
        ),
        key=lambda row: (float(row["mean_rank"]), str(row["model_id"])),
    )


def _aggregate_rank_matches(actual: list[Any], expected: list[dict[str, Any]]) -> bool:
    actual_by_model = _rank_by_model(actual)
    expected_by_model = _rank_by_model(expected)
    if set(actual_by_model) != set(expected_by_model):
        return False
    for model_id, expected_row in expected_by_model.items():
        actual_row = actual_by_model[model_id]
        actual_rank = _finite_mean_rank(actual_row)
        expected_rank = _finite_mean_rank(expected_row)
        if actual_rank is None or expected_rank is None or abs(actual_rank - expected_rank) > 1e-9:
            return False
    return True


def _rank_by_model(rows: list[Any]) -> dict[str, dict[str, Any]]:
    by_model = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("model_id") is None:
            continue
        by_model[str(row["model_id"])] = row
    return by_model


def _finite_mean_rank(row: dict[str, Any]) -> float | None:
    return _finite_float(row.get("mean_rank", row.get("rank")))


def _finite_rank(row: dict[str, Any]) -> float | None:
    return _finite_float(row.get("rank", row.get("mean_rank")))


def _finite_float(value: Any) -> float | None:
    if _finite_number(value):
        return float(value)
    return None


def _seed_aggregate_ci_violations(payload: dict[str, Any], aggregate_rank: list[Any]) -> list[str]:
    rows = payload.get("seed_aggregate")
    if not isinstance(rows, list):
        return ["paper mode requires seed_aggregate CI rows"]
    violations: list[str] = []
    ranked_models = {
        str(row.get("model_id"))
        for row in aggregate_rank
        if isinstance(row, dict) and row.get("model_id") is not None
    }
    for model_id in sorted(ranked_models):
        candidates = [
            row for row in rows
            if isinstance(row, dict)
            and str(row.get("model_id")) == model_id
            and row.get("metric") == "mse"
        ]
        if not any(_valid_seed_aggregate_ci_row(row) for row in candidates):
            violations.append(f"seed_aggregate:{model_id}:mse lacks finite cross-seed CI summary")
    return violations


def _valid_seed_aggregate_ci_row(row: dict[str, Any]) -> bool:
    n_seeds = row.get("n_seeds")
    return (
        _finite_number(row.get("mean"))
        and _finite_number(row.get("ci_low"))
        and _finite_number(row.get("ci_high"))
        and isinstance(n_seeds, int)
        and not isinstance(n_seeds, bool)
        and n_seeds >= len(CANONICAL_REQUIRED_SEEDS)
    )


def _has_finite_ci(metrics: dict[str, Any], metric: str) -> bool:
    low = metrics.get(f"{metric}_ci_low")
    high = metrics.get(f"{metric}_ci_high")
    return _finite_number(low) and _finite_number(high)


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float, np.floating)) and np.isfinite(float(value))
