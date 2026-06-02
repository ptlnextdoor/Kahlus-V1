from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from neurotwin.contracts.paper_mode import CANONICAL_REQUIRED_SEEDS
from neurotwin.scoring.metrics import bootstrap_ci, rank_models


CONCRETE_SEED_CONTAINER_KEYS = ("runs", "seed_results", "per_seed_results", "baseline_runs", "per_seed_baselines")


@dataclass(frozen=True)
class ModelMetricRecord:
    task_id: str
    model_id: str
    metric: str
    value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SeedRunRecord:
    seed: int | None
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True)
class AggregateRankRecord:
    model_id: str
    mean_rank: float
    tasks_ranked: int
    std_rank: float | None = None
    n_seeds: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model_id": self.model_id,
            "mean_rank": self.mean_rank,
            "tasks_ranked": self.tasks_ranked,
        }
        if self.std_rank is not None:
            payload["std_rank"] = self.std_rank
        if self.n_seeds is not None:
            payload["n_seeds"] = self.n_seeds
        return payload


@dataclass(frozen=True)
class SeedAggregateRecord:
    task_id: str
    model_id: str
    metric: str
    mean: float
    std: float
    ci_low: float
    ci_high: float
    n_seeds: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PaperModeEvidence:
    seed_results: tuple[dict[str, Any], ...]
    aggregate_rank: tuple[AggregateRankRecord, ...]
    expected_aggregate_rank: tuple[AggregateRankRecord, ...]
    seed_aggregate: tuple[SeedAggregateRecord, ...]
    required_seeds: tuple[int, ...] = CANONICAL_REQUIRED_SEEDS
    require_ci: bool = True
    seed_records: tuple[SeedRunRecord, ...] = field(default_factory=tuple)
    audit_payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class PaperModeGateResult:
    passed: bool
    violations: tuple[str, ...]
    warnings: tuple[str, ...]
    checked: tuple[str, ...]
    required_seeds: tuple[int, ...]
    observed_seeds: tuple[int, ...]
    require_ci: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_paper_mode_evidence(
    seed_results: Sequence[Mapping[str, Any]],
    required_seeds: Sequence[int] = CANONICAL_REQUIRED_SEEDS,
    require_ci: bool = True,
) -> PaperModeEvidence:
    records = tuple(_seed_run_record(dict(result)) for result in seed_results)
    result_payloads = tuple(record.payload for record in records)
    aggregate_rank = tuple(AggregateRankRecord(**row) for row in aggregate_seed_ranks(result_payloads))
    seed_aggregate = tuple(SeedAggregateRecord(**row) for row in aggregate_seed_metrics(result_payloads))
    return PaperModeEvidence(
        seed_results=result_payloads,
        aggregate_rank=aggregate_rank,
        expected_aggregate_rank=aggregate_rank,
        seed_aggregate=seed_aggregate,
        required_seeds=tuple(int(seed) for seed in required_seeds),
        require_ci=bool(require_ci),
        seed_records=records,
    )


def evidence_to_json(evidence: PaperModeEvidence) -> dict[str, Any]:
    return {
        "aggregate": {
            "selection_metric": "mse",
            "higher_is_better": False,
            "aggregate_rank": [row.to_dict() for row in evidence.aggregate_rank],
        },
        "seed_results": [dict(result) for result in evidence.seed_results],
        "seed_aggregate": [row.to_dict() for row in evidence.seed_aggregate],
        "paper_mode_contract": {
            "required_seeds": list(evidence.required_seeds),
            "observed_seeds": [record.seed for record in evidence.seed_records if record.seed is not None],
            "require_ci": evidence.require_ci,
            "gate_status": "not_run",
        },
    }


def evidence_from_json(
    payload: PaperModeEvidence | Mapping[str, Any],
    required_seeds: Sequence[int] = CANONICAL_REQUIRED_SEEDS,
    require_ci: bool = True,
) -> PaperModeEvidence:
    if isinstance(payload, PaperModeEvidence):
        return payload
    seed_records = tuple(_seed_run_record(record) for record in _iter_concrete_seed_records(payload))
    expected = tuple(AggregateRankRecord(**row) for row in aggregate_seed_ranks([record.payload for record in seed_records]))
    aggregate_rank = tuple(_aggregate_rank_from_payload(payload))
    seed_aggregate = tuple(_seed_aggregate_from_payload(payload))
    return PaperModeEvidence(
        seed_results=tuple(record.payload for record in seed_records),
        aggregate_rank=aggregate_rank,
        expected_aggregate_rank=expected,
        seed_aggregate=seed_aggregate,
        required_seeds=tuple(int(seed) for seed in required_seeds),
        require_ci=bool(require_ci),
        seed_records=seed_records,
        audit_payload=_top_level_audit_payload(payload),
    )


def validate_paper_mode_evidence(
    evidence: PaperModeEvidence | Mapping[str, Any],
    audit_report: Any | None = None,
    require_ci: bool = True,
) -> PaperModeGateResult:
    evidence = evidence_from_json(evidence, require_ci=require_ci)
    metric_evidence_check = "ci_summaries" if require_ci else "metric_evidence_without_ci"
    checked = ("eval_audit", "baseline_rankings", "required_seeds", metric_evidence_check)
    violations: list[str] = []
    warnings: list[str] = []

    audit_payload = _audit_payload(audit_report, evidence)
    if audit_payload is None:
        violations.append("paper mode requires a prepared eval audit payload")
    elif not bool(audit_payload.get("passed")):
        details = audit_payload.get("violations")
        suffix = f": {details}" if details else ""
        violations.append(f"prepared eval audit did not pass{suffix}")

    if not evidence.aggregate_rank:
        violations.append("baseline aggregate_rank is empty")
    elif not evidence.expected_aggregate_rank:
        violations.append("paper mode requires per-seed ranked baseline evidence")
    elif not _aggregate_rank_matches(evidence.aggregate_rank, evidence.expected_aggregate_rank):
        violations.append("baseline aggregate_rank does not match per-seed ranked baseline evidence")

    observed_seeds, seed_evidence_violations = _observed_seeds(evidence.seed_records, require_ci=bool(require_ci))
    missing = tuple(seed for seed in evidence.required_seeds if seed not in observed_seeds)
    if missing:
        violations.append(
            "paper mode requires seeds "
            + ",".join(str(seed) for seed in evidence.required_seeds)
            + "; missing "
            + ",".join(str(seed) for seed in missing)
        )
    for seed in missing:
        violations.extend(seed_evidence_violations.get(seed, ()))

    if require_ci:
        violations.extend(_seed_aggregate_ci_violations(evidence))

    return PaperModeGateResult(
        passed=not violations,
        violations=tuple(violations),
        warnings=tuple(warnings),
        checked=checked,
        required_seeds=evidence.required_seeds,
        observed_seeds=observed_seeds,
        require_ci=bool(require_ci),
    )


def aggregate_seed_ranks(seed_results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ranks_by_model: dict[str, list[tuple[str, float]]] = {}
    for result_idx, result in enumerate(seed_results):
        seed_key = _seed_key(result, result_idx)
        tasks = result.get("tasks", {})
        if not isinstance(tasks, dict):
            continue
        for task_payload in tasks.values():
            if not isinstance(task_payload, dict):
                continue
            ranking = task_payload.get("ranking", [])
            if not isinstance(ranking, list):
                continue
            for row in ranking:
                if not isinstance(row, dict):
                    continue
                model_id = row.get("model_id")
                if model_id is None:
                    continue
                rank = _finite_rank(row)
                if rank is not None:
                    ranks_by_model.setdefault(str(model_id), []).append((seed_key, rank))
    return sorted(
        (
            {
                "model_id": model_id,
                "mean_rank": float(np.mean(ranks)),
                "std_rank": float(np.std(ranks)),
                "tasks_ranked": len(ranks),
                "n_seeds": len(seed_keys),
            }
            for model_id, values in ranks_by_model.items()
            for ranks, seed_keys in [([rank for _, rank in values], {seed_key for seed_key, _ in values})]
            if ranks
        ),
        key=lambda row: (float(row["mean_rank"]), str(row["model_id"])),
    )


def aggregate_seed_metrics(seed_results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values: dict[tuple[str, str, str], list[float]] = {}
    for result in seed_results:
        tasks = result.get("tasks", {})
        if not isinstance(tasks, dict):
            continue
        for task_id, task_payload in tasks.items():
            if not isinstance(task_payload, dict):
                continue
            metrics_by_model = task_payload.get("metrics_by_model", {})
            if isinstance(metrics_by_model, dict):
                for model_id, metrics in metrics_by_model.items():
                    if isinstance(metrics, dict):
                        _collect_metric_values(values, str(task_id), str(model_id), metrics)
            task_metrics = task_payload.get("metrics", {})
            if isinstance(task_metrics, dict):
                _collect_metric_values(values, str(task_id), "task_metric", task_metrics)
    rows: list[dict[str, Any]] = []
    for (task_id, model_id, metric), metric_values in sorted(values.items()):
        arr = np.asarray(metric_values, dtype=float)
        if arr.size == 0 or not np.isfinite(arr).all():
            continue
        if arr.size == 1:
            ci_low = ci_high = float(arr[0])
        else:
            ci_low, ci_high = bootstrap_ci(arr, seed=_stable_metric_seed(task_id, model_id, metric), n_boot=200)
        rows.append(
            {
                "task_id": task_id,
                "model_id": model_id,
                "metric": metric,
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "n_seeds": int(arr.size),
            }
        )
    return rows


def aggregated_seed_tasks(
    representative_payload: Mapping[str, Any],
    seed_aggregate: Sequence[SeedAggregateRecord | Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    representative_tasks = representative_payload.get("tasks", {})
    if not isinstance(representative_tasks, dict):
        representative_tasks = {}

    rows_by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in seed_aggregate:
        payload = row.to_dict() if isinstance(row, SeedAggregateRecord) else row
        task_id = payload.get("task_id")
        if task_id is not None:
            rows_by_task.setdefault(str(task_id), []).append(payload)

    tasks: dict[str, dict[str, Any]] = {}
    for task_id, rows in sorted(rows_by_task.items()):
        representative = representative_tasks.get(task_id, {})
        if not isinstance(representative, dict):
            representative = {}
        metrics: dict[str, float] = {}
        metrics_by_model: dict[str, dict[str, float]] = {}
        for row in rows:
            model_id = row.get("model_id")
            metric = row.get("metric")
            if model_id is None or metric is None:
                continue
            metric_name = str(metric)
            mean = _finite_row_float(row, "mean")
            ci_low = _finite_row_float(row, "ci_low")
            ci_high = _finite_row_float(row, "ci_high")
            if mean is None or ci_low is None or ci_high is None:
                continue
            target = metrics if str(model_id) == "task_metric" else metrics_by_model.setdefault(str(model_id), {})
            target[metric_name] = mean
            target[f"{metric_name}_ci_low"] = ci_low
            target[f"{metric_name}_ci_high"] = ci_high

        ranking = [
            {"model_id": row.model_id, "metric": row.metric, "value": row.value, "rank": row.rank}
            for row in rank_models(metrics_by_model, metric="mse", higher_is_better=False)
        ] if metrics_by_model and all("mse" in model_metrics for model_metrics in metrics_by_model.values()) else []
        tasks[task_id] = {
            "status": "seed_aggregated",
            "source_modality": representative.get("source_modality"),
            "target_modality": representative.get("target_modality"),
            "metrics": metrics,
            "metrics_by_model": metrics_by_model,
            "ranking": ranking,
            "failures": [],
            "notes": [
                "aggregate across seed_results; per-seed concrete results are stored in seed_results",
                *_representative_notes(representative),
            ],
        }
    return tasks


def _aggregate_rank_from_payload(payload: Mapping[str, Any]) -> list[AggregateRankRecord]:
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict):
        return []
    rows = aggregate.get("aggregate_rank")
    if not isinstance(rows, list):
        return []
    records = []
    for row in rows:
        if not isinstance(row, dict) or row.get("model_id") is None:
            continue
        rank = _finite_mean_rank(row)
        if rank is None:
            continue
        records.append(
            AggregateRankRecord(
                model_id=str(row["model_id"]),
                mean_rank=rank,
                tasks_ranked=int(row.get("tasks_ranked", row.get("n_tasks", 0)) or 0),
                std_rank=_finite_float(row.get("std_rank")),
                n_seeds=_optional_int(row.get("n_seeds")),
            )
        )
    return records


def _seed_aggregate_from_payload(payload: Mapping[str, Any]) -> list[SeedAggregateRecord]:
    rows = payload.get("seed_aggregate")
    if not isinstance(rows, list):
        return []
    records = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        mean = _finite_row_float(row, "mean")
        std = _finite_row_float(row, "std")
        ci_low = _finite_row_float(row, "ci_low")
        ci_high = _finite_row_float(row, "ci_high")
        n_seeds = _optional_int(row.get("n_seeds"))
        if (
            row.get("task_id") is None
            or row.get("model_id") is None
            or row.get("metric") is None
        ):
            continue
        records.append(
            SeedAggregateRecord(
                task_id=str(row["task_id"]),
                model_id=str(row["model_id"]),
                metric=str(row["metric"]),
                mean=float("nan") if mean is None else mean,
                std=float("nan") if std is None else std,
                ci_low=float("nan") if ci_low is None else ci_low,
                ci_high=float("nan") if ci_high is None else ci_high,
                n_seeds=0 if n_seeds is None else n_seeds,
            )
        )
    return records


def _seed_run_record(payload: Mapping[str, Any]) -> SeedRunRecord:
    record = dict(payload)
    return SeedRunRecord(seed=_coerce_seed(record.get("seed")), payload=record)


def _iter_concrete_seed_records(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
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


def _observed_seeds(
    seed_records: Sequence[SeedRunRecord],
    require_ci: bool = True,
) -> tuple[tuple[int, ...], dict[int, tuple[str, ...]]]:
    observed: set[int] = set()
    violations: dict[int, list[str]] = {}
    for record in seed_records:
        seed = record.seed
        if seed not in CANONICAL_REQUIRED_SEEDS:
            continue
        passed, record_violations = _seed_record_has_ranked_baseline_evidence(record.payload, require_ci=require_ci)
        if passed:
            observed.add(seed)
        elif seed is not None:
            violations.setdefault(seed, []).extend(record_violations)
    return tuple(sorted(observed)), {seed: tuple(values) for seed, values in violations.items()}


def _seed_record_has_ranked_baseline_evidence(
    record: Mapping[str, Any],
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


def _seed_aggregate_ci_violations(evidence: PaperModeEvidence) -> list[str]:
    if not evidence.seed_aggregate:
        return ["paper mode requires seed_aggregate CI rows"]
    violations: list[str] = []
    rows = [row.to_dict() for row in evidence.seed_aggregate]
    ranked_models = {row.model_id for row in evidence.aggregate_rank}
    for model_id in sorted(ranked_models):
        candidates = [
            row for row in rows
            if str(row.get("model_id")) == model_id
            and row.get("metric") == "mse"
        ]
        if not any(_valid_seed_aggregate_ci_row(row, evidence.required_seeds) for row in candidates):
            violations.append(f"seed_aggregate:{model_id}:mse lacks finite cross-seed CI summary")
    return violations


def _valid_seed_aggregate_ci_row(row: Mapping[str, Any], required_seeds: Sequence[int]) -> bool:
    n_seeds = row.get("n_seeds")
    return (
        _finite_number(row.get("mean"))
        and _finite_number(row.get("ci_low"))
        and _finite_number(row.get("ci_high"))
        and isinstance(n_seeds, int)
        and not isinstance(n_seeds, bool)
        and n_seeds >= len(required_seeds)
    )


def _aggregate_rank_matches(actual: Sequence[AggregateRankRecord], expected: Sequence[AggregateRankRecord]) -> bool:
    actual_by_model = {row.model_id: row for row in actual}
    expected_by_model = {row.model_id: row for row in expected}
    if set(actual_by_model) != set(expected_by_model):
        return False
    for model_id, expected_row in expected_by_model.items():
        actual_rank = actual_by_model[model_id].mean_rank
        expected_rank = expected_row.mean_rank
        if abs(actual_rank - expected_rank) > 1e-9:
            return False
    return True


def _audit_payload(audit_report: Any | None, evidence: PaperModeEvidence) -> dict[str, Any] | None:
    if audit_report is not None:
        if hasattr(audit_report, "to_dict"):
            audit_report = audit_report.to_dict()
        elif hasattr(audit_report, "passed"):
            audit_report = {
                "passed": bool(getattr(audit_report, "passed")),
                "violations": tuple(getattr(audit_report, "violations", ())),
            }
        return audit_report if isinstance(audit_report, dict) else None
    if evidence.audit_payload is not None:
        return evidence.audit_payload
    for result in evidence.seed_results:
        for key in ("eval_audit", "prepared_eval_audit", "audit"):
            value = result.get(key)
            if isinstance(value, dict):
                return value
    return None


def _top_level_audit_payload(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    for key in ("eval_audit", "prepared_eval_audit", "audit"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return None


def _collect_metric_values(
    values: dict[tuple[str, str, str], list[float]],
    task_id: str,
    model_id: str,
    metrics: Mapping[str, Any],
) -> None:
    for metric, value in metrics.items():
        if str(metric).endswith(("_ci_low", "_ci_high")):
            continue
        if _finite_number(value):
            values.setdefault((task_id, model_id, str(metric)), []).append(float(value))


def _representative_notes(task_payload: Mapping[str, Any]) -> list[str]:
    notes = task_payload.get("notes", [])
    return [str(note) for note in notes] if isinstance(notes, list) else []


def _seed_key(result: Mapping[str, Any], fallback_index: int) -> str:
    value = result.get("seed", fallback_index)
    return str(value)


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


def normalize_seed_tuple(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    normalized: list[int] = []
    for item in value:
        seed = _coerce_seed(item)
        if seed is None:
            return None
        normalized.append(seed)
    return tuple(normalized)


def _finite_rank(row: Mapping[str, Any]) -> float | None:
    return _finite_float(row.get("rank", row.get("mean_rank")))


def _finite_mean_rank(row: Mapping[str, Any]) -> float | None:
    return _finite_float(row.get("mean_rank", row.get("rank")))


def _finite_float(value: Any) -> float | None:
    if _finite_number(value):
        return float(value)
    return None


def _finite_row_float(row: Mapping[str, Any], key: str) -> float | None:
    return _finite_float(row.get(key))


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _has_finite_ci(metrics: Mapping[str, Any], metric: str) -> bool:
    low = metrics.get(f"{metric}_ci_low")
    high = metrics.get(f"{metric}_ci_high")
    return _finite_number(low) and _finite_number(high)


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float, np.floating)) and np.isfinite(float(value))


def _stable_metric_seed(task_id: str, model_id: str, metric: str) -> int:
    return sum(ord(char) for char in f"{task_id}:{model_id}:{metric}") % 1_000_003
