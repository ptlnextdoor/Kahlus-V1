from __future__ import annotations

from pathlib import Path

import numpy as np

from neurotwin.benchmarks.baseline_suite import run_synthetic_baseline_suite
from neurotwin.benchmarks.tasks import (
    TaskResult,
    run_cross_modal_translation_task,
    run_dataset_site_generalization_task,
    run_future_forecasting_task,
    run_masked_reconstruction_task,
    run_subject_adaptation_task,
)
from neurotwin.repro import write_json


def run_neural_translation_v1_synthetic(seed: int = 0, out_dir: str | Path | None = None) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    eeg = rng.normal(size=(64, 8)).astype(np.float32)
    fmri = eeg[:, :4] @ rng.normal(size=(4, 6)).astype(np.float32) + 0.05 * rng.normal(size=(64, 6))
    results = [
        run_future_forecasting_task(eeg, history=8, horizon=1),
        run_masked_reconstruction_task(eeg, mask_fraction=0.2, seed=seed),
        run_cross_modal_translation_task({"eeg": eeg, "fmri": fmri}, source="eeg", target="fmri"),
        run_subject_adaptation_task(eeg[:16], eeg[16:32]),
        run_dataset_site_generalization_task(eeg[:32], eeg[32:], source_name="synthetic_a", target_name="synthetic_b"),
    ]
    payload = {result.task_id: _task_result_to_dict(result) for result in results}
    payload["scope"] = {
        "status": "synthetic-only",
        "notes": ["validates benchmark plumbing, not scientific performance"],
    }
    payload["baseline_suite"] = run_synthetic_baseline_suite(seed=seed)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "metrics.json", payload)
        write_json(out / "baseline_suite.json", payload["baseline_suite"])
        write_json(out / "summary.json", {"suite": "neural_translation_v1", "synthetic_only": True})
    return payload


def format_neural_translation_v1_report(payload: dict[str, object]) -> str:
    lines = ["# NeuroTwin Neural Translation V1", "", "scope=synthetic-only", ""]
    for task_id, result in payload.items():
        if task_id in {"scope", "baseline_suite"}:
            continue
        if not isinstance(result, dict):
            continue
        lines.append(f"## {task_id}")
        lines.append(f"status={result['status']}")
        metrics = result.get("metrics", {})
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                lines.append(f"{key}={value}")
        notes = result.get("notes", [])
        if notes:
            lines.append("notes=" + "; ".join(str(note) for note in notes))
        lines.append("")
    baseline_suite = payload.get("baseline_suite", {})
    if isinstance(baseline_suite, dict):
        aggregate = baseline_suite.get("aggregate", {})
        if isinstance(aggregate, dict):
            lines.append("## local_baseline_suite")
            lines.append("status=synthetic-only")
            lines.append("selection_metric=mse")
            for row in aggregate.get("aggregate_rank", []):
                if isinstance(row, dict):
                    lines.append(
                        f"ranked_model={row.get('model_id')} mean_rank={row.get('mean_rank')} tasks={row.get('tasks_ranked')}"
                    )
            lines.append("")
    lines.append("Synthetic/debug results are plumbing checks, not scientific evidence.")
    return "\n".join(lines)


def _task_result_to_dict(result: TaskResult) -> dict[str, object]:
    return {
        "status": result.status,
        "metrics": result.metrics,
        "notes": result.notes,
    }
