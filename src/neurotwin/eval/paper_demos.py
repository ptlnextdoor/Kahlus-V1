from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from neurotwin.adapters.synthetic import make_synthetic_event_batches, make_synthetic_recordings
from neurotwin.benchmarks.baseline_suite import run_supervised_window_tasks
from neurotwin.contracts.paper_mode import CANONICAL_REQUIRED_SEEDS
from neurotwin.data.event_io import event_manifest_summary, load_event_batches
from neurotwin.data.manifest_io import load_split_manifest
from neurotwin.data.prepared_tasks import (
    build_future_forecasting_task_from_windows,
    first_prepared_modality_with_splits,
    prepared_windows_by_split,
)
from neurotwin.data.schemas import NeuralEventBatch
from neurotwin.data.split_manifest import SplitManifest, build_split_manifest
from neurotwin.data.windows import WindowSpec, batch_to_windows
from neurotwin.repro import write_json


@dataclass(frozen=True)
class PaperDemoConfig:
    dataset: str = "synthetic"
    event_manifest: str | Path | None = None
    split_manifest: str | Path | None = None
    out_dir: str | Path | None = None
    window_length: int = 8
    stride: int = 8
    seed: int = 0
    seeds: tuple[int, ...] | None = None
    train_steps: int = 1


def run_leakage_demo(config: PaperDemoConfig) -> dict[str, Any]:
    seeds = _resolved_seeds(config)
    _validate_demo_config(config, command_name="leakage-demo")
    seed_results = [_run_seed_with_failures(seed, config, _leakage_demo_seed_payload) for seed in seeds]
    completed = [result for result in seed_results if result.get("status") == "completed"]
    source = _source_from_results(completed, config)
    single_seed = len(seeds) == 1
    payload = {
        "demo": "segment_vs_subject_split",
        "dataset": source["dataset"],
        "event_manifest": source.get("event_manifest"),
        "split_manifest": source.get("split_manifest"),
        "window_length": config.window_length,
        "stride": config.stride,
        "seeds": list(seeds),
        "requested_seeds": list(seeds),
        "observed_seeds": [int(result["seed"]) for result in completed],
        "scientific_claim_allowed": False,
        "evidence_status": _evidence_status(seed_results),
        "claim_control": "leakage demo is evidence about split validity, not model superiority",
        "seed_results": seed_results,
        "seed_aggregate": _aggregate_leakage_seed_results(completed),
        "paper_demo_gate": _paper_demo_gate(seed_results),
        "interpretation": _aggregate_leakage_interpretation(completed),
    }
    if single_seed:
        payload["seed"] = seeds[0]
        payload["comparisons"] = completed[0].get("comparisons", []) if completed else []
    elif completed:
        payload["representative_seed_result"] = _representative_seed_result(completed[0])
    _write_demo_payload(config.out_dir, "leakage_demo", payload)
    return payload


def format_leakage_demo(payload: dict[str, Any]) -> str:
    lines = [
        "eval_leakage_demo=True",
        f"dataset={payload.get('dataset')}",
        "requested_seeds=" + ",".join(str(seed) for seed in payload.get("requested_seeds", [])),
        "observed_seeds=" + ",".join(str(seed) for seed in payload.get("observed_seeds", [])),
        f"evidence_status={payload.get('evidence_status')}",
        f"scientific_claim_allowed={payload.get('scientific_claim_allowed')}",
        f"paper_demo_gate_passed={_nested_bool(payload, 'paper_demo_gate', 'passed')}",
        f"interpretation={payload.get('interpretation')}",
    ]
    for row in payload.get("seed_aggregate", []):
        if isinstance(row, dict):
            lines.append(
                "seed_aggregate="
                f"{row.get('split_id')} metric={row.get('metric')} mean={row.get('mean')} "
                f"std={row.get('std')} ci95=[{row.get('ci_low')},{row.get('ci_high')}] n={row.get('n_seeds')}"
            )
    comparison_prefix = "split_result" if payload.get("evidence_status") == "single_seed_non_paper" else "representative_split_result"
    for row in _representative_comparisons(payload):
        lines.append(
            f"{comparison_prefix}="
            f"{row.get('split_id')} status={row.get('status')} "
            f"mse={row.get('mse')} mae={row.get('mae')} "
            f"train_subjects={row.get('train_subjects')} test_subjects={row.get('test_subjects')} "
            f"subject_overlap={row.get('subject_overlap')} "
            f"scientific_claim_allowed={row.get('scientific_claim_allowed')}"
        )
    for failure in _seed_failures(payload):
        lines.append(f"seed_failure={failure.get('seed')} error={failure.get('error')}")
    return "\n".join(lines)


def run_identity_probe(config: PaperDemoConfig) -> dict[str, Any]:
    seeds = _resolved_seeds(config)
    _validate_demo_config(config, command_name="identity-probe")
    seed_results = [_run_seed_with_failures(seed, config, _identity_probe_seed_payload) for seed in seeds]
    completed = [result for result in seed_results if result.get("status") == "completed"]
    source = _source_from_results(completed, config)
    representative_probe = completed[0].get("window_split_probe", {}) if completed else {}
    single_seed = len(seeds) == 1
    payload = {
        "probe": "subject_identity_from_neural_windows",
        "dataset": source["dataset"],
        "event_manifest": source.get("event_manifest"),
        "split_manifest": source.get("split_manifest"),
        "window_length": config.window_length,
        "stride": config.stride,
        "seeds": list(seeds),
        "requested_seeds": list(seeds),
        "observed_seeds": [int(result["seed"]) for result in completed],
        "modality": representative_probe.get("modality") if isinstance(representative_probe, dict) else None,
        "scientific_claim_allowed": False,
        "evidence_status": _evidence_status(seed_results),
        "claim_control": "identity probe quantifies confounding risk and is not a model-performance claim",
        "identity_confounding_risk": _aggregate_identity_risk(completed),
        "seed_results": seed_results,
        "seed_aggregate": _aggregate_identity_seed_results(completed),
        "paper_demo_gate": _paper_demo_gate(seed_results),
    }
    if single_seed:
        payload["seed"] = seeds[0]
        payload["window_split_probe"] = representative_probe
        payload["heldout_split_subject_overlap"] = completed[0].get("heldout_split_subject_overlap", {}) if completed else {}
    elif completed:
        payload["representative_seed_result"] = _representative_seed_result(completed[0])
    _write_demo_payload(config.out_dir, "identity_probe", payload)
    return payload


def format_identity_probe(payload: dict[str, Any]) -> str:
    lines = [
        "eval_identity_probe=True",
        f"dataset={payload.get('dataset')}",
        "requested_seeds=" + ",".join(str(seed) for seed in payload.get("requested_seeds", [])),
        "observed_seeds=" + ",".join(str(seed) for seed in payload.get("observed_seeds", [])),
        f"evidence_status={payload.get('evidence_status')}",
        f"modality={payload.get('modality')}",
        f"scientific_claim_allowed={payload.get('scientific_claim_allowed')}",
        f"paper_demo_gate_passed={_nested_bool(payload, 'paper_demo_gate', 'passed')}",
        f"identity_confounding_risk={payload.get('identity_confounding_risk')}",
    ]
    for row in payload.get("seed_aggregate", []):
        if isinstance(row, dict):
            lines.append(
                "seed_aggregate="
                f"metric={row.get('metric')} mean={row.get('mean')} std={row.get('std')} "
                f"ci95=[{row.get('ci_low')},{row.get('ci_high')}] n={row.get('n_seeds')}"
            )
    probe = _representative_probe(payload)
    if probe:
        prefix = "" if payload.get("evidence_status") == "single_seed_non_paper" else "representative_"
        lines.extend(
            [
                f"{prefix}window_split_accuracy={probe.get('accuracy')}",
                f"{prefix}chance_accuracy={probe.get('chance_accuracy')}",
                f"{prefix}subjects={probe.get('subjects')}",
            ]
        )
    overlap = _representative_overlap(payload)
    if overlap:
        prefix = "" if payload.get("evidence_status") == "single_seed_non_paper" else "representative_"
        lines.append(
            f"{prefix}heldout_subject_overlap="
            f"{overlap.get('count')} train={overlap.get('train_subjects')} test={overlap.get('test_subjects')}"
        )
    for failure in _seed_failures(payload):
        lines.append(f"seed_failure={failure.get('seed')} error={failure.get('error')}")
    return "\n".join(lines)


def _leakage_demo_seed_payload(config: PaperDemoConfig) -> dict[str, Any]:
    batches, split, source = _load_demo_inputs(config)
    correct = _subject_split_forecast_metrics(batches, split, config)
    negative = _bad_segment_split_forecast_metrics(batches, config)
    return {
        "dataset": source["dataset"],
        "event_manifest": source.get("event_manifest"),
        "split_manifest": source.get("split_manifest"),
        "comparisons": [negative, correct],
        "interpretation": _leakage_interpretation(negative, correct),
    }


def _identity_probe_seed_payload(config: PaperDemoConfig) -> dict[str, Any]:
    batches, split, source = _load_demo_inputs(config)
    windows = _modality_windows(batches, config, preferred="eeg")
    probe = _window_identity_probe(windows, seed=config.seed)
    return {
        "dataset": source["dataset"],
        "event_manifest": source.get("event_manifest"),
        "split_manifest": source.get("split_manifest"),
        "window_split_probe": probe,
        "heldout_split_subject_overlap": _split_subject_overlap(split),
        "identity_confounding_risk": _identity_risk(probe),
    }


def _run_seed_with_failures(seed: int, config: PaperDemoConfig, runner: Any) -> dict[str, Any]:
    seed_config = replace(config, seed=int(seed), seeds=None)
    try:
        payload = runner(seed_config)
    except Exception as exc:  # noqa: BLE001 - seed failures are artifact data.
        return {"seed": int(seed), "status": "failed", "error": str(exc)}
    return {"seed": int(seed), "status": "completed", **payload}


def _load_demo_inputs(config: PaperDemoConfig) -> tuple[list[NeuralEventBatch], SplitManifest, dict[str, Any]]:
    if config.event_manifest or config.split_manifest:
        if config.event_manifest is None or config.split_manifest is None:
            raise ValueError("--event-manifest and --split-manifest must be provided together")
        batches = load_event_batches(config.event_manifest)
        split = load_split_manifest(config.split_manifest)
        summary = event_manifest_summary(config.event_manifest)
        return batches, split, {
            "dataset": str(summary.get("dataset") or config.dataset),
            "event_manifest": str(config.event_manifest),
            "split_manifest": str(config.split_manifest),
        }
    if config.dataset != "synthetic":
        raise ValueError(f"{config.dataset} demos require prepared --event-manifest and --split-manifest")
    records = make_synthetic_recordings(n_subjects=8, sessions_per_subject=2, modalities=("eeg",))
    batches = make_synthetic_event_batches(n_subjects=8, sessions_per_subject=2, modalities=("eeg",), n_time=64)
    split = build_split_manifest(records, policy="subject", seed=config.seed)
    return batches, split, {"dataset": "synthetic"}


def _subject_split_forecast_metrics(
    batches: list[NeuralEventBatch],
    split: SplitManifest,
    config: PaperDemoConfig,
) -> dict[str, Any]:
    windows_by_split = prepared_windows_by_split(batches, split, window_length=config.window_length, stride=config.stride)
    modality = first_prepared_modality_with_splits(windows_by_split) or "eeg"
    train = [window for window in windows_by_split["train"] if window.modality == modality]
    test = [window for window in windows_by_split["test"] if window.modality == modality]
    task = build_future_forecasting_task_from_windows(
        windows_by_split,
        notes=(f"{modality} next-state forecasting under held-out {split.policy} split",),
    )
    metrics = _baseline_forecast_metrics(task, seed=config.seed, train_steps=config.train_steps)
    subjects = _train_test_subjects(train, test)
    return {
        "split_id": "correct_subject_split" if split.policy == "subject" else f"correct_{split.policy}_split",
        "status": "claim_eligible_split_candidate",
        "negative_control": False,
        "scientific_claim_allowed": False,
        "notes": ["Still requires real data, required seeds, and a passed paper-mode gate before claims."],
        **subjects,
        **metrics,
    }


def _bad_segment_split_forecast_metrics(
    batches: list[NeuralEventBatch],
    config: PaperDemoConfig,
) -> dict[str, Any]:
    windows = _modality_windows(batches, config, preferred="eeg")
    train, test = _leaky_window_split_by_subject(windows, seed=config.seed)
    windows_by_split = {"train": train, "val": [], "test": test}
    task = build_future_forecasting_task_from_windows(
        windows_by_split,
        notes=("intentional bad segment split with subject identity overlap",),
    )
    metrics = _baseline_forecast_metrics(task, seed=config.seed, train_steps=config.train_steps)
    subjects = _train_test_subjects(train, test)
    return {
        "split_id": "bad_segment_split",
        "status": "negative_control",
        "negative_control": True,
        "scientific_claim_allowed": False,
        "notes": ["Intentionally leaks subject identity by allowing windows from the same subject in train and test."],
        **subjects,
        **metrics,
    }


def _baseline_forecast_metrics(task: Any, seed: int, train_steps: int) -> dict[str, Any]:
    if task is None:
        return {"status_detail": "skipped", "reason": "need train/test windows with at least two timepoints"}
    payload = run_supervised_window_tasks(
        (task,),
        seed=seed,
        train_steps=train_steps,
        scope_status="leakage-demo-diagnostic",
        scope_notes=("Diagnostic negative-control runner. Do not use as model-superiority evidence.",),
        model_ids=("linear_ridge",),
    )
    task_payload = payload["tasks"].get(task.task_id, {})
    metrics_by_model = task_payload.get("metrics_by_model", {}) if isinstance(task_payload, dict) else {}
    model_id = _selected_metric_model(metrics_by_model, task_payload)
    metrics = metrics_by_model.get(model_id, {}) if isinstance(metrics_by_model, dict) and model_id else {}
    result = {
        "status_detail": task_payload.get("status", "skipped") if isinstance(task_payload, dict) else "skipped",
        "model_id": model_id or "missing",
        "train_windows": int(task.x_train.shape[0]),
        "test_windows": int(task.x_test.shape[0]),
        "metrics_by_model": metrics_by_model,
        "ranking": task_payload.get("ranking", []) if isinstance(task_payload, dict) else [],
        "baseline_failures": task_payload.get("failures", []) if isinstance(task_payload, dict) else [],
    }
    if isinstance(metrics, dict):
        result.update(metrics)
    return result


def _selected_metric_model(metrics_by_model: Any, task_payload: Any) -> str | None:
    if isinstance(metrics_by_model, dict) and "linear_ridge" in metrics_by_model:
        return "linear_ridge"
    ranking = task_payload.get("ranking", []) if isinstance(task_payload, dict) else []
    if isinstance(ranking, list):
        for row in ranking:
            if isinstance(row, dict) and row.get("model_id") in metrics_by_model:
                return str(row["model_id"])
    if isinstance(metrics_by_model, dict) and metrics_by_model:
        return sorted(str(model_id) for model_id in metrics_by_model)[0]
    return None


def _modality_windows(
    batches: list[NeuralEventBatch],
    config: PaperDemoConfig,
    preferred: str,
) -> list[NeuralEventBatch]:
    modalities = sorted({batch.modality for batch in batches})
    if not modalities:
        raise ValueError("no event batches available for paper demo")
    modality = preferred if preferred in modalities else modalities[0]
    spec = WindowSpec(length=config.window_length, stride=config.stride)
    windows: list[NeuralEventBatch] = []
    for batch in batches:
        if batch.modality == modality:
            windows.extend(batch_to_windows(batch, spec))
    return windows


def _leaky_window_split_by_subject(windows: list[NeuralEventBatch], seed: int) -> tuple[list[NeuralEventBatch], list[NeuralEventBatch]]:
    rng = np.random.default_rng(seed)
    by_subject: dict[str, list[NeuralEventBatch]] = {}
    for window in windows:
        by_subject.setdefault(window.subject_id, []).append(window)
    train: list[NeuralEventBatch] = []
    test: list[NeuralEventBatch] = []
    for subject_windows in by_subject.values():
        order = rng.permutation(len(subject_windows))
        cut = max(1, len(order) // 2)
        train.extend(subject_windows[int(idx)] for idx in order[:cut])
        test.extend(subject_windows[int(idx)] for idx in order[cut:])
    return train, test


def _train_test_subjects(train: list[NeuralEventBatch], test: list[NeuralEventBatch]) -> dict[str, Any]:
    train_subjects = {window.subject_id for window in train}
    test_subjects = {window.subject_id for window in test}
    overlap = sorted(train_subjects & test_subjects)
    return {
        "train_subjects": len(train_subjects),
        "test_subjects": len(test_subjects),
        "subject_overlap": len(overlap),
        "overlapping_subject_ids": overlap,
    }


def _window_identity_probe(windows: list[NeuralEventBatch], seed: int) -> dict[str, Any]:
    if not windows:
        return {"status": "skipped", "reason": "no windows available"}
    train, test = _leaky_window_split_by_subject(windows, seed=seed)
    if not train or not test:
        return {"status": "skipped", "reason": "need train/test windows"}
    centroids = {}
    for subject_id in sorted({window.subject_id for window in train}):
        rows = [_identity_features(window) for window in train if window.subject_id == subject_id]
        if rows:
            centroids[subject_id] = np.mean(np.asarray(rows, dtype=np.float64), axis=0)
    predictions: list[str] = []
    labels: list[str] = []
    for window in test:
        if not centroids:
            continue
        features = _identity_features(window)
        pred = min(centroids, key=lambda subject_id: float(np.mean((features - centroids[subject_id]) ** 2)))
        predictions.append(pred)
        labels.append(window.subject_id)
    correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    subjects = sorted({window.subject_id for window in windows})
    accuracy = correct / max(1, len(labels))
    chance = 1.0 / max(1, len(subjects))
    return {
        "status": "completed",
        "split_id": "bad_window_split_subject_identity_probe",
        "status_control": "negative_control",
        "modality": windows[0].modality,
        "model_id": "nearest_subject_centroid",
        "accuracy": float(accuracy),
        "chance_accuracy": float(chance),
        "test_windows": len(labels),
        "subjects": len(subjects),
        "train_subjects": len({window.subject_id for window in train}),
        "test_subjects": len({window.subject_id for window in test}),
        "subject_overlap": len({window.subject_id for window in train} & {window.subject_id for window in test}),
    }


def _identity_features(window: NeuralEventBatch) -> np.ndarray:
    signal = np.asarray(window.signal, dtype=np.float64)
    return np.concatenate([signal.mean(axis=0), signal.std(axis=0)])


def _split_subject_overlap(split: SplitManifest) -> dict[str, Any]:
    train_subjects = {record.subject_id for record in split.train}
    test_subjects = {record.subject_id for record in split.test}
    overlap = sorted(train_subjects & test_subjects)
    return {
        "train_subjects": len(train_subjects),
        "test_subjects": len(test_subjects),
        "count": len(overlap),
        "subject_ids": overlap,
        "heldout_subject_probe_applicable": bool(overlap),
    }


def _resolved_seeds(config: PaperDemoConfig) -> tuple[int, ...]:
    if config.seeds is None:
        return (int(config.seed),)
    if not config.seeds:
        raise ValueError("--seeds must include at least one seed")
    return tuple(int(seed) for seed in config.seeds)


def _validate_demo_config(config: PaperDemoConfig, command_name: str) -> None:
    if bool(config.event_manifest) != bool(config.split_manifest):
        raise ValueError("--event-manifest and --split-manifest must be provided together")
    if config.dataset != "synthetic" and (config.event_manifest is None or config.split_manifest is None):
        raise ValueError(f"{config.dataset} {command_name} requires prepared --event-manifest and --split-manifest")


def _paper_demo_gate(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    requested = tuple(int(result["seed"]) for result in seed_results)
    observed = tuple(int(result["seed"]) for result in seed_results if result.get("status") == "completed")
    violations: list[str] = []
    failures = [result for result in seed_results if result.get("status") != "completed"]
    if requested != CANONICAL_REQUIRED_SEEDS:
        violations.append("paper diagnostics require canonical seeds 0,1,2")
    if observed != requested:
        violations.append("not all requested seeds completed")
    for failure in failures:
        violations.append(f"seed {failure.get('seed')} failed: {failure.get('error')}")
    return {
        "passed": not violations,
        "required_seeds": list(CANONICAL_REQUIRED_SEEDS),
        "requested_seeds": list(requested),
        "observed_seeds": list(observed),
        "violations": violations,
        "warnings": ["diagnostic artifacts are never direct model-superiority claims"],
        "claim_allowed": False,
    }


def _evidence_status(seed_results: list[dict[str, Any]]) -> str:
    failures = [result for result in seed_results if result.get("status") != "completed"]
    if failures:
        return "multi_seed_failed" if len(seed_results) > 1 else "single_seed_failed"
    requested = tuple(int(result["seed"]) for result in seed_results)
    if len(requested) == 1:
        return "single_seed_non_paper"
    if requested == CANONICAL_REQUIRED_SEEDS:
        return "canonical_multi_seed_diagnostic"
    return "noncanonical_multi_seed_diagnostic"


def _aggregate_leakage_seed_results(seed_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[float]] = {}
    for result in seed_results:
        for row in result.get("comparisons", []):
            if not isinstance(row, dict):
                continue
            split_id = str(row.get("split_id", "unknown"))
            for metric in ("mse", "mae", "pearsonr", "spearmanr", "r2", "train_windows", "test_windows", "subject_overlap"):
                value = row.get(metric)
                if isinstance(value, (int, float, np.floating)) and np.isfinite(float(value)):
                    by_key.setdefault((split_id, metric), []).append(float(value))
    aggregates = []
    for (split_id, metric), values in sorted(by_key.items()):
        aggregates.append({"split_id": split_id, "metric": metric, **_numeric_summary(values)})
    return aggregates


def _aggregate_identity_seed_results(seed_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values_by_metric: dict[str, list[float]] = {}
    for result in seed_results:
        probe = result.get("window_split_probe", {})
        if not isinstance(probe, dict):
            continue
        for metric in ("accuracy", "chance_accuracy", "test_windows", "subjects", "train_subjects", "test_subjects", "subject_overlap"):
            value = probe.get(metric)
            if isinstance(value, (int, float, np.floating)) and np.isfinite(float(value)):
                values_by_metric.setdefault(metric, []).append(float(value))
    return [{"metric": metric, **_numeric_summary(values)} for metric, values in sorted(values_by_metric.items())]


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    half_width = float(1.96 * std / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "n_seeds": int(arr.size),
    }


def _aggregate_leakage_interpretation(seed_results: list[dict[str, Any]]) -> str:
    if not seed_results:
        return "failed"
    aggregate = _aggregate_leakage_seed_results(seed_results)
    mse_by_split = {
        row["split_id"]: row["mean"]
        for row in aggregate
        if row.get("metric") == "mse" and isinstance(row.get("mean"), (int, float))
    }
    bad_mse = mse_by_split.get("bad_segment_split")
    correct_mse = next((value for split_id, value in mse_by_split.items() if split_id.startswith("correct_")), None)
    if bad_mse is None or correct_mse is None:
        return "incomplete"
    if float(bad_mse) < float(correct_mse):
        return "bad_segment_split_looks_better_and_is_not_claim_eligible"
    return "bad_segment_split_did_not_improve_here_but_remains_not_claim_eligible"


def _leakage_interpretation(negative: dict[str, Any], correct: dict[str, Any]) -> str:
    if negative.get("status_detail") != "completed" or correct.get("status_detail") != "completed":
        return "incomplete"
    bad_mse = float(negative["mse"])
    correct_mse = float(correct["mse"])
    if bad_mse < correct_mse:
        return "bad_segment_split_looks_better_and_is_not_claim_eligible"
    return "bad_segment_split_did_not_improve_here_but_remains_not_claim_eligible"


def _aggregate_identity_risk(seed_results: list[dict[str, Any]]) -> str:
    risks = [str(result.get("identity_confounding_risk")) for result in seed_results]
    if not risks:
        return "unknown"
    if "high" in risks:
        return "high"
    if "elevated" in risks:
        return "elevated"
    if all(risk == "low" for risk in risks):
        return "low"
    return "unknown"


def _identity_risk(probe: dict[str, Any]) -> str:
    if probe.get("status") != "completed":
        return "unknown"
    accuracy = float(probe.get("accuracy", 0.0))
    chance = float(probe.get("chance_accuracy", 0.0))
    if accuracy >= 0.5 and accuracy >= max(0.0, chance) * 2.0:
        return "high"
    if accuracy > chance:
        return "elevated"
    return "low"


def _source_from_results(seed_results: list[dict[str, Any]], config: PaperDemoConfig) -> dict[str, Any]:
    for result in seed_results:
        dataset = result.get("dataset")
        if dataset:
            return {
                "dataset": str(dataset),
                "event_manifest": result.get("event_manifest"),
                "split_manifest": result.get("split_manifest"),
            }
    return {
        "dataset": config.dataset,
        "event_manifest": str(config.event_manifest) if config.event_manifest is not None else None,
        "split_manifest": str(config.split_manifest) if config.split_manifest is not None else None,
    }


def _representative_seed_result(seed_result: dict[str, Any]) -> dict[str, Any]:
    return dict(seed_result)


def _representative_comparisons(payload: dict[str, Any]) -> list[dict[str, Any]]:
    comparisons = payload.get("comparisons")
    if isinstance(comparisons, list):
        return [row for row in comparisons if isinstance(row, dict)]
    representative = payload.get("representative_seed_result")
    if not isinstance(representative, dict):
        return []
    representative_comparisons = representative.get("comparisons")
    if not isinstance(representative_comparisons, list):
        return []
    return [row for row in representative_comparisons if isinstance(row, dict)]


def _representative_probe(payload: dict[str, Any]) -> dict[str, Any]:
    probe = payload.get("window_split_probe")
    if isinstance(probe, dict):
        return probe
    representative = payload.get("representative_seed_result")
    if not isinstance(representative, dict):
        return {}
    representative_probe = representative.get("window_split_probe")
    return representative_probe if isinstance(representative_probe, dict) else {}


def _representative_overlap(payload: dict[str, Any]) -> dict[str, Any]:
    overlap = payload.get("heldout_split_subject_overlap")
    if isinstance(overlap, dict):
        return overlap
    representative = payload.get("representative_seed_result")
    if not isinstance(representative, dict):
        return {}
    representative_overlap = representative.get("heldout_split_subject_overlap")
    return representative_overlap if isinstance(representative_overlap, dict) else {}


def _nested_bool(payload: dict[str, Any], parent: str, child: str) -> Any:
    value = payload.get(parent)
    return value.get(child) if isinstance(value, dict) else None


def _seed_failures(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("seed_results", [])
    if not isinstance(results, list):
        return []
    return [result for result in results if isinstance(result, dict) and result.get("status") != "completed"]


def _write_demo_payload(out_dir: str | Path | None, stem: str, payload: dict[str, Any]) -> None:
    if out_dir is None:
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"{stem}.json", payload)
    write_json(out / f"{stem.upper()}.json", payload)
