from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from neurotwin.benchmarks.baseline_suite import (
    PreparedAggregateRankPayload,
    PreparedBaselineSuitePayload,
    PreparedPaperModePayload,
    SeedAggregatePayload,
)
from neurotwin.benchmarks.prepared_suite import run_prepared_baseline_suite
from neurotwin.contracts.paper_mode import CANONICAL_REQUIRED_SEEDS
from neurotwin.data.event_io import event_manifest_summary
from neurotwin.data.prepared_tasks import PreparedSuiteConfig
from neurotwin.eval.paper_contracts import (
    aggregate_seed_metrics,
    aggregate_seed_ranks,
    aggregated_seed_tasks,
    build_paper_mode_evidence,
    evidence_to_json,
)
from neurotwin.eval.paper_gate import PaperModeGateReport
from neurotwin.repro import write_json


def run_prepared_baseline_suite_multi_seed(
    config: PreparedSuiteConfig,
    seeds: tuple[int, ...] | list[int] = CANONICAL_REQUIRED_SEEDS,
    out_dir: str | Path | None = None,
) -> PreparedPaperModePayload:
    """Run prepared baselines for multiple seeds and assemble paper-mode artifacts."""

    seed_values = tuple(int(seed) for seed in seeds)
    seed_results: list[PreparedBaselineSuitePayload] = []
    for seed in seed_values:
        seed_payload = run_prepared_baseline_suite(
            PreparedSuiteConfig(
                event_manifest=config.event_manifest,
                split_manifest=config.split_manifest,
                window_length=config.window_length,
                stride=config.stride,
                seed=seed,
                train_steps=config.train_steps,
                require_ci=config.require_ci,
                max_windows_per_split=config.max_windows_per_split,
            ),
            out_dir=None,
        )
        seed_payload["seed"] = seed
        seed_payload["seeds"] = [seed]
        seed_results.append(seed_payload)

    payload = _merge_seed_payloads(config, seed_values, seed_results)
    if out_dir is not None:
        write_prepared_paper_mode_artifacts(payload, out_dir)
    return payload


def write_prepared_paper_mode_artifacts(
    payload: PreparedPaperModePayload,
    out_dir: str | Path,
    gate: PaperModeGateReport | None = None,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "prepared_baseline_suite.json", payload)
    write_json(out / "seed_aggregate.json", payload.get("seed_aggregate", []))
    (out / "seed_aggregate.csv").write_text(
        _seed_aggregate_csv(payload.get("seed_aggregate", [])),
        encoding="utf-8",
    )
    write_json(out / "baseline_failures.json", payload.get("baseline_failures", []))
    if gate is not None:
        write_json(out / "paper_mode_gate.json", gate.to_dict())


def _merge_seed_payloads(
    config: PreparedSuiteConfig,
    seeds: tuple[int, ...],
    seed_results: list[PreparedBaselineSuitePayload],
) -> PreparedPaperModePayload:
    first: PreparedBaselineSuitePayload | dict[str, object] = seed_results[0] if seed_results else {}
    evidence = build_paper_mode_evidence(
        seed_results,
        required_seeds=CANONICAL_REQUIRED_SEEDS,
        require_ci=config.require_ci,
    )
    evidence_payload = evidence_to_json(evidence)
    seed_aggregate = [row.to_dict() for row in evidence.seed_aggregate]
    tasks = aggregated_seed_tasks(first, evidence.seed_aggregate)
    failures: list[dict[str, str]] = []
    for result in seed_results:
        raw_failures = result.get("baseline_failures", [])
        if isinstance(raw_failures, list):
            for failure in raw_failures:
                if isinstance(failure, dict):
                    failures.append({str(key): str(value) for key, value in failure.items()})
    return cast(
        PreparedPaperModePayload,
        {
            "scope": first.get("scope", {"status": "prepared-data", "notes": []}),
            "tasks": tasks,
            "aggregate": evidence_payload["aggregate"],
            "seed": int(seeds[0]) if seeds else int(config.seed),
            "seeds": [int(seed) for seed in seeds],
            "seed_results": seed_results,
            "seed_aggregate": seed_aggregate,
            "representative_seed_tasks": first.get("tasks", {}),
            "prepared_data": first.get(
                "prepared_data",
                {
                    "event_manifest": str(config.event_manifest),
                    "split_manifest": str(config.split_manifest),
                    "event_summary": event_manifest_summary(config.event_manifest),
                    "window_length": config.window_length,
                    "stride": config.stride,
                    "max_windows_per_split": config.max_windows_per_split,
                    "skipped_tasks": [],
                },
            ),
            "paper_mode_contract": {
                "required_seeds": list(CANONICAL_REQUIRED_SEEDS),
                "observed_seeds": [int(seed) for seed in seeds],
                "require_ci": bool(config.require_ci),
                "gate_status": "not_run",
            },
            "baseline_catalog": first.get("baseline_catalog", []),
            "baseline_failures": failures,
        },
    )


def _aggregate_seed_tasks(
    representative_payload: PreparedBaselineSuitePayload | dict[str, object],
    seed_aggregate: list[SeedAggregatePayload],
) -> dict[str, Any]:
    return aggregated_seed_tasks(representative_payload, seed_aggregate)


def _aggregate_seed_ranks(seed_results: list[PreparedBaselineSuitePayload]) -> list[PreparedAggregateRankPayload]:
    return cast(list[PreparedAggregateRankPayload], aggregate_seed_ranks(seed_results))


def _aggregate_seed_metrics(seed_results: list[PreparedBaselineSuitePayload]) -> list[SeedAggregatePayload]:
    return cast(list[SeedAggregatePayload], aggregate_seed_metrics(seed_results))


def _seed_aggregate_csv(rows: object) -> str:
    header = ("task_id", "model_id", "metric", "mean", "std", "ci_low", "ci_high", "n_seeds")
    lines = [",".join(header)]
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                lines.append(",".join(_csv_cell(row.get(key, "")) for key in header))
    return "\n".join(lines) + "\n"


def _csv_cell(value: object) -> str:
    text = str(value)
    if any(char in text for char in (",", "\"", "\n")):
        return "\"" + text.replace("\"", "\"\"") + "\""
    return text
