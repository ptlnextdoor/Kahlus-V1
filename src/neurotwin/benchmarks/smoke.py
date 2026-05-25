from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurotwin.adapters.synthetic import make_synthetic_recordings
from neurotwin.data.leakage import check_manifest_leakage
from neurotwin.data.split_manifest import build_split_manifest
from neurotwin.eval.metrics import RankingRow, mse, pearsonr, rank_models


@dataclass(frozen=True)
class SmokeBenchmarkResult:
    suite: str
    split_policy: str
    leakage_passed: bool
    metrics: dict[str, dict[str, float]]
    ranking: list[RankingRow]
    notes: tuple[str, ...]


def run_translation_smoke(seed: int = 0) -> SmokeBenchmarkResult:
    """Run deterministic synthetic-only scoring to verify benchmark plumbing."""

    records = make_synthetic_recordings(
        n_subjects=12,
        sessions_per_subject=2,
        modalities=("fmri", "eeg", "meg"),
        sites=("site-a", "site-b", "site-c"),
        datasets=("synthetic_a", "synthetic_b", "synthetic_c"),
    )
    manifest = build_split_manifest(records, policy="subject", seed=seed)
    leakage = check_manifest_leakage(manifest, keys=("subject_id",))

    rng = np.random.default_rng(seed)
    y_true = rng.normal(size=(64, 12))
    base_signal = y_true + 0.05 * rng.normal(size=y_true.shape)
    predictions = {
        "transformer": base_signal + 0.30 * rng.normal(size=y_true.shape),
        "mamba_ssm": base_signal + 0.24 * rng.normal(size=y_true.shape),
        "modality_specialist": base_signal + 0.22 * rng.normal(size=y_true.shape),
        "neurotwin": base_signal + 0.20 * rng.normal(size=y_true.shape),
    }
    metrics = {
        model_id: {
            "mse": mse(y_true, pred),
            "pearsonr": pearsonr(y_true, pred),
        }
        for model_id, pred in predictions.items()
    }
    ranking = rank_models(metrics, metric="mse", higher_is_better=False)
    return SmokeBenchmarkResult(
        suite="translation_smoke",
        split_policy=manifest.policy,
        leakage_passed=leakage.passed,
        metrics=metrics,
        ranking=ranking,
        notes=(
            "synthetic-only benchmark: validates plumbing, not scientific performance",
            "real acceptance requires public-data manifests and upstream baselines",
        ),
    )


def format_smoke_results(result: SmokeBenchmarkResult) -> str:
    lines = [
        "# NeuroTwin Synthetic Smoke Results",
        "",
        f"suite={result.suite}",
        f"split_policy={result.split_policy}",
        f"leakage_passed={result.leakage_passed}",
        "scope=synthetic-only",
        "",
        "| aggregate_rank | model | mse | pearsonr |",
        "| --- | --- | --- | --- |",
    ]
    for row in result.ranking:
        model_metrics = result.metrics[row.model_id]
        lines.append(
            f"| {row.rank} | {row.model_id} | {model_metrics['mse']:.6f} | {model_metrics['pearsonr']:.6f} |"
        )
    lines.extend(["", "Notes:"])
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)
