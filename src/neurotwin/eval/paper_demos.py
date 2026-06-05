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
    task: str = "future_state_forecasting"
    event_manifest: str | Path | None = None
    split_manifest: str | Path | None = None
    out_dir: str | Path | None = None
    window_length: int = 8
    stride: int = 8
    seed: int = 0
    seeds: tuple[int, ...] | None = None
    train_steps: int = 1


def run_leakage_demo(config: PaperDemoConfig) -> dict[str, Any]:
    task = _normalized_demo_task(config.task)
    if task == "motor_imagery_classification":
        return run_classification_leakage_demo(config)
    if task != "future_state_forecasting":
        raise ValueError(f"unsupported leakage-demo task: {config.task}")
    seeds = _resolved_seeds(config)
    _validate_demo_config(config, command_name="leakage-demo")
    seed_results = [_run_seed_with_failures(seed, config, _leakage_demo_seed_payload) for seed in seeds]
    completed = [result for result in seed_results if result.get("status") == "completed"]
    source = _source_from_results(completed, config)
    single_seed = len(seeds) == 1
    payload = {
        "demo": "segment_vs_subject_split",
        "task": task,
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


def run_classification_leakage_demo(config: PaperDemoConfig) -> dict[str, Any]:
    seeds = _resolved_seeds(config)
    _validate_demo_config(config, command_name="leakage-demo")
    task_warning = _deprecated_task_alias_warning(config.task)
    seed_results = [_run_seed_with_failures(seed, config, _classification_leakage_seed_payload) for seed in seeds]
    completed = [result for result in seed_results if result.get("status") == "completed"]
    source = _source_from_results(completed, config)
    single_seed = len(seeds) == 1
    claim_gate = _classification_claim_gate(seed_results)
    payload = {
        "demo": "motor_imagery_classification_segment_vs_subject_split",
        "task": "motor_imagery_classification",
        "dataset": source["dataset"],
        "event_manifest": source.get("event_manifest"),
        "split_manifest": source.get("split_manifest"),
        "window_length": config.window_length,
        "stride": config.stride,
        "seeds": list(seeds),
        "requested_seeds": list(seeds),
        "observed_seeds": [int(result["seed"]) for result in completed],
        "models": ["linear_ridge_classifier"],
        "scientific_claim_allowed": False,
        "evidence_status": _evidence_status(seed_results),
        "claim_control": "classification leakage demo is evidence about split validity, not model superiority",
        "deprecation_warning": task_warning,
        "seed_results": seed_results,
        "seed_aggregate": _aggregate_classification_seed_results(completed),
        "inflation_summary": _aggregate_classification_inflation(completed),
        "paper_demo_gate": _paper_demo_gate(seed_results),
        "claim_gate": claim_gate,
        "figure_data": _classification_figure_data(completed, claim_gate),
        "identity_probe_link": {
            "artifact": "identity_probe.json",
            "note": "Run nt eval identity-probe with the same manifests/seeds for subject recoverability evidence.",
        },
        "interpretation": _aggregate_classification_interpretation(completed),
    }
    if single_seed:
        payload["seed"] = seeds[0]
        payload["comparisons"] = completed[0].get("comparisons", []) if completed else []
    elif completed:
        payload["representative_seed_result"] = _representative_seed_result(completed[0])
    _write_classification_demo_artifacts(config.out_dir, payload)
    return payload


def format_leakage_demo(payload: dict[str, Any]) -> str:
    if payload.get("task") == "motor_imagery_classification":
        return format_classification_leakage_demo(payload)
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


def format_classification_leakage_demo(payload: dict[str, Any]) -> str:
    lines = [
        "eval_leakage_demo=True",
        "eval_classification_leakage_demo=True",
        f"dataset={payload.get('dataset')}",
        f"task={payload.get('task')}",
        "requested_seeds=" + ",".join(str(seed) for seed in payload.get("requested_seeds", [])),
        "observed_seeds=" + ",".join(str(seed) for seed in payload.get("observed_seeds", [])),
        f"evidence_status={payload.get('evidence_status')}",
        f"scientific_claim_allowed={payload.get('scientific_claim_allowed')}",
        f"paper_demo_gate_passed={_nested_bool(payload, 'paper_demo_gate', 'passed')}",
        f"claim_gate_bad_split_allowed={_nested_bool(payload, 'claim_gate', 'bad_split_claim_allowed')}",
        f"interpretation={payload.get('interpretation')}",
    ]
    if payload.get("deprecation_warning"):
        lines.append(f"deprecation_warning={payload.get('deprecation_warning')}")
    for row in payload.get("seed_aggregate", []):
        if isinstance(row, dict):
            lines.append(
                "classification_seed_aggregate="
                f"{row.get('split_id')} {row.get('model_id')} metric={row.get('metric')} mean={row.get('mean')} "
                f"std={row.get('std')} ci95=[{row.get('ci_low')},{row.get('ci_high')}] n={row.get('n_seeds')}"
            )
    comparison_prefix = "classification_split_result" if payload.get("evidence_status") == "single_seed_non_paper" else "representative_classification_split_result"
    for row in _representative_comparisons(payload):
        lines.append(
            f"{comparison_prefix}="
            f"{row.get('split_id')} status={row.get('status')} "
            f"accuracy={row.get('accuracy')} balanced_accuracy={row.get('balanced_accuracy')} "
            f"f1={row.get('f1')} auroc={row.get('auroc')} "
            f"train_subjects={row.get('train_subjects')} test_subjects={row.get('test_subjects')} "
            f"subject_overlap={row.get('subject_overlap')} "
            f"scientific_claim_allowed={row.get('scientific_claim_allowed')}"
        )
    for row in payload.get("inflation_summary", []):
        if isinstance(row, dict):
            lines.append(
                "classification_inflation="
                f"metric={row.get('metric')} difference={row.get('difference')} ratio={row.get('ratio')}"
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


def _classification_leakage_seed_payload(config: PaperDemoConfig) -> dict[str, Any]:
    batches, split, source = _load_demo_inputs(config)
    windows = _labeled_classification_windows(batches, split, config)
    negative = _bad_segment_split_classification_metrics(windows, seed=config.seed)
    correct = _subject_heldout_classification_metrics(windows, seed=config.seed)
    return {
        "dataset": source["dataset"],
        "event_manifest": source.get("event_manifest"),
        "split_manifest": source.get("split_manifest"),
        "class_labels": sorted({str(row["label"]) for row in windows}),
        "comparisons": [negative, correct],
        "inflation": _classification_inflation(negative, correct),
        "interpretation": _classification_interpretation(negative, correct),
    }


def _labeled_classification_windows(
    batches: list[NeuralEventBatch],
    split: SplitManifest,
    config: PaperDemoConfig,
) -> list[dict[str, Any]]:
    records_by_id = {record.record_id: record for record in split.all_records}
    split_by_record = _split_record_keys(split)
    labeled = []
    for window in _modality_windows(batches, config, preferred="eeg"):
        record_id = _record_id(window)
        split_name = split_by_record.get(record_id)
        if split_name not in {"train", "test"}:
            continue
        label = _classification_label(records_by_id.get(record_id), window)
        if label is None:
            continue
        labeled.append({"window": window, "label": str(label), "split": split_name, "record_id": record_id})
    if len({row["label"] for row in labeled}) < 2:
        raise ValueError("motor_imagery_classification requires at least two class labels from split_manifest stimulus_id")
    if not any(row["split"] == "train" for row in labeled) or not any(row["split"] == "test" for row in labeled):
        raise ValueError("motor_imagery_classification requires labeled train and test windows")
    return labeled


def _classification_label(record: Any, window: NeuralEventBatch) -> str | None:
    if record is not None and getattr(record, "stimulus_id", None) not in (None, ""):
        return str(record.stimulus_id)
    value = window.metadata.get("stimulus_id")
    if value not in (None, ""):
        return str(value)
    alignment = window.metadata.get("stimulus_alignment")
    if isinstance(alignment, dict) and alignment.get("stimulus_id") not in (None, ""):
        return str(alignment["stimulus_id"])
    return None


def _bad_segment_split_classification_metrics(windows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    train, test = _leaky_classification_split_by_subject(windows, seed=seed)
    return _classification_split_metrics(
        train,
        test,
        split_id="bad_segment_split",
        status="negative_control",
        negative_control=True,
        notes=("Random window split ignoring held-out subject boundaries; not claim eligible.",),
        seed=seed,
    )


def _subject_heldout_classification_metrics(windows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    train = [row for row in windows if row["split"] == "train"]
    test = [row for row in windows if row["split"] == "test"]
    return _classification_split_metrics(
        train,
        test,
        split_id="correct_subject_heldout",
        status="valid_split_candidate",
        negative_control=False,
        notes=("Subject-held-out split candidate; still not a standalone model-superiority claim.",),
        seed=seed,
    )


def _leaky_classification_split_by_subject(windows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for row in windows:
        by_subject.setdefault(str(row["window"].subject_id), []).append(row)
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for subject_windows in by_subject.values():
        order = rng.permutation(len(subject_windows))
        cut = max(1, len(order) // 2)
        train.extend(subject_windows[int(idx)] for idx in order[:cut])
        test.extend(subject_windows[int(idx)] for idx in order[cut:])
    return train, test


def _classification_split_metrics(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    *,
    split_id: str,
    status: str,
    negative_control: bool,
    notes: tuple[str, ...],
    seed: int,
) -> dict[str, Any]:
    subjects = _classification_subjects(train, test)
    class_counts = _classification_class_counts(train, test)
    payload: dict[str, Any] = {
        "split_id": split_id,
        "status": status,
        "negative_control": bool(negative_control),
        "scientific_claim_allowed": False,
        "model_id": "linear_ridge_classifier",
        "notes": list(notes),
        "train_windows": len(train),
        "test_windows": len(test),
        **subjects,
        **class_counts,
    }
    if not train or not test:
        return {**payload, "status_detail": "skipped", "reason": "need train and test windows"}
    train_labels = {str(row["label"]) for row in train}
    test_labels = {str(row["label"]) for row in test}
    class_labels = sorted(train_labels | test_labels)
    if len(train_labels) < 2 or len(test_labels) < 2:
        return {**payload, "status_detail": "skipped", "reason": "need at least two classes in train and test windows"}
    x_train = _classification_feature_matrix(train)
    x_test = _classification_feature_matrix(test)
    y_train = np.asarray([class_labels.index(str(row["label"])) for row in train], dtype=np.int64)
    y_test = np.asarray([class_labels.index(str(row["label"])) for row in test], dtype=np.int64)
    prediction, probabilities = _fit_linear_ridge_classifier(x_train, y_train, x_test, n_classes=len(class_labels))
    metrics = _classification_metrics(y_test, prediction, probabilities, class_labels=class_labels, seed=seed)
    return {
        **payload,
        "status_detail": "completed",
        "classes": class_labels,
        "metrics_by_model": {"linear_ridge_classifier": metrics},
        "ranking": [
            {
                "model_id": "linear_ridge_classifier",
                "metric": "balanced_accuracy",
                "value": metrics["balanced_accuracy"],
                "rank": 1,
            }
        ],
        **metrics,
    }


def _classification_feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([_classification_features(row["window"]) for row in rows], dtype=np.float32)


def _classification_features(window: NeuralEventBatch) -> np.ndarray:
    signal = np.asarray(window.signal, dtype=np.float32)
    signal = np.nan_to_num(signal, nan=0.0, posinf=1e6, neginf=-1e6)
    mean = np.mean(signal, axis=0)
    std = np.std(signal, axis=0)
    log_var = np.log1p(np.var(signal, axis=0))
    features = np.concatenate([mean, std, log_var]).astype(np.float32)
    return np.clip(np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)


def _fit_linear_ridge_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    n_classes: int,
    alpha: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    mean = np.mean(x_train, axis=0, keepdims=True)
    std = np.std(x_train, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    train = (x_train - mean) / std
    test = (x_test - mean) / std
    train = np.clip(np.nan_to_num(train, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)
    test = np.clip(np.nan_to_num(test, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)
    train_aug = np.concatenate([train, np.ones((train.shape[0], 1), dtype=np.float64)], axis=1)
    test_aug = np.concatenate([test, np.ones((test.shape[0], 1), dtype=np.float64)], axis=1)
    target = np.zeros((train_aug.shape[0], n_classes), dtype=np.float64)
    target[np.arange(train_aug.shape[0]), y_train] = 1.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        lhs = train_aug.T @ train_aug + alpha * np.eye(train_aug.shape[1], dtype=np.float64)
        rhs = train_aug.T @ target
    lhs = np.nan_to_num(lhs, nan=0.0, posinf=1e12, neginf=-1e12)
    rhs = np.nan_to_num(rhs, nan=0.0, posinf=1e12, neginf=-1e12)
    try:
        weights = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(lhs) @ rhs
    scores = test_aug @ weights
    return np.argmax(scores, axis=1).astype(np.int64), _softmax(scores)


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    *,
    class_labels: list[str],
    seed: int,
) -> dict[str, float]:
    values = _classification_metric_values(y_true, y_pred, probabilities, class_labels=class_labels)
    metrics: dict[str, float] = {}
    for metric, value in values.items():
        if value is None or not np.isfinite(float(value)):
            continue
        metrics[metric] = float(value)
        low, high = _bootstrap_classification_ci(y_true, y_pred, probabilities, class_labels=class_labels, metric=metric, seed=seed)
        metrics[f"{metric}_ci_low"] = low
        metrics[f"{metric}_ci_high"] = high
    return metrics


def _classification_metric_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    *,
    class_labels: list[str],
) -> dict[str, float | None]:
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": _balanced_accuracy(y_true, y_pred),
        "f1": _macro_f1(y_true, y_pred),
        "auroc": _binary_auroc(y_true, probabilities, class_labels=class_labels),
    }


def _bootstrap_classification_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    *,
    class_labels: list[str],
    metric: str,
    seed: int,
    n_boot: int = 500,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed + 101)
    if y_true.size == 0:
        return 0.0, 0.0
    values = []
    for _ in range(n_boot):
        idx = rng.choice(y_true.size, size=y_true.size, replace=True)
        metrics = _classification_metric_values(y_true[idx], y_pred[idx], probabilities[idx], class_labels=class_labels)
        value = metrics.get(metric)
        if value is not None and np.isfinite(float(value)):
            values.append(float(value))
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for label in sorted(set(int(value) for value in y_true.tolist())):
        mask = y_true == label
        if np.any(mask):
            recalls.append(float(np.mean(y_pred[mask] == label)))
    return float(np.mean(recalls)) if recalls else 0.0


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = sorted(set(int(value) for value in y_true.tolist()) | set(int(value) for value in y_pred.tolist()))
    scores = []
    for label in labels:
        tp = float(np.sum((y_true == label) & (y_pred == label)))
        fp = float(np.sum((y_true != label) & (y_pred == label)))
        fn = float(np.sum((y_true == label) & (y_pred != label)))
        denom = (2.0 * tp) + fp + fn
        scores.append(0.0 if denom == 0.0 else (2.0 * tp) / denom)
    return float(np.mean(scores)) if scores else 0.0


def _binary_auroc(y_true: np.ndarray, probabilities: np.ndarray, *, class_labels: list[str]) -> float | None:
    if len(class_labels) != 2 or probabilities.shape[1] != 2:
        return None
    positive_scores = probabilities[:, 1]
    positives = y_true == 1
    n_pos = int(np.sum(positives))
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _rankdata_average(positive_scores)
    pos_rank_sum = float(np.sum(ranks[positives]))
    return float((pos_rank_sum - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg))


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=1, keepdims=True)
    return exp / np.maximum(denom, 1e-12)


def _classification_subjects(train: list[dict[str, Any]], test: list[dict[str, Any]]) -> dict[str, Any]:
    train_subjects = {row["window"].subject_id for row in train}
    test_subjects = {row["window"].subject_id for row in test}
    overlap = sorted(train_subjects & test_subjects)
    return {
        "train_subjects": len(train_subjects),
        "test_subjects": len(test_subjects),
        "subject_overlap": len(overlap),
        "overlapping_subject_ids": overlap,
    }


def _classification_class_counts(train: list[dict[str, Any]], test: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "train_class_counts": _label_counts(train),
        "test_class_counts": _label_counts(test),
        "class_count": len(set(_label_counts(train)) | set(_label_counts(test))),
    }


def _label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row["label"])
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _classification_inflation(negative: dict[str, Any], correct: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for metric in ("accuracy", "balanced_accuracy", "f1", "auroc"):
        bad_value = negative.get(metric)
        correct_value = correct.get(metric)
        if not isinstance(bad_value, (int, float, np.floating)) or not isinstance(correct_value, (int, float, np.floating)):
            continue
        ratio = None if np.isclose(float(correct_value), 0.0) else float(bad_value) / float(correct_value)
        rows.append(
            {
                "metric": metric,
                "difference": float(bad_value) - float(correct_value),
                "ratio": ratio,
                "bad_segment_split": float(bad_value),
                "correct_subject_heldout": float(correct_value),
            }
        )
    return rows


def _classification_interpretation(negative: dict[str, Any], correct: dict[str, Any]) -> str:
    if negative.get("status_detail") != "completed" or correct.get("status_detail") != "completed":
        return "incomplete"
    bad_value = float(negative.get("balanced_accuracy", 0.0))
    correct_value = float(correct.get("balanced_accuracy", 0.0))
    if bad_value > correct_value:
        return "bad_segment_split_inflates_classification_and_is_not_claim_eligible"
    return "bad_segment_split_did_not_inflate_classification_here_but_remains_not_claim_eligible"


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
            "dataset": _demo_dataset_label(summary, config.dataset),
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


def _record_id(batch: NeuralEventBatch) -> str:
    return str(batch.metadata.get("record_id") or batch.metadata.get("source_record_id"))


def _split_record_keys(split: SplitManifest) -> dict[str, str]:
    keys = {}
    for split_name in ("train", "val", "test"):
        for record in getattr(split, split_name):
            keys[record.record_id] = split_name
    return keys


def _resolved_seeds(config: PaperDemoConfig) -> tuple[int, ...]:
    if config.seeds is None:
        return (int(config.seed),)
    if not config.seeds:
        raise ValueError("--seeds must include at least one seed")
    return tuple(int(seed) for seed in config.seeds)


def _validate_demo_config(config: PaperDemoConfig, command_name: str) -> None:
    _normalized_demo_task(config.task)
    if bool(config.event_manifest) != bool(config.split_manifest):
        raise ValueError("--event-manifest and --split-manifest must be provided together")
    if config.dataset != "synthetic" and (config.event_manifest is None or config.split_manifest is None):
        raise ValueError(f"{config.dataset} {command_name} requires prepared --event-manifest and --split-manifest")


def _normalized_demo_task(task: str) -> str:
    normalized = str(task).strip().lower().replace("-", "_")
    aliases = {
        "forecasting": "future_state_forecasting",
        "future": "future_state_forecasting",
        "future_state_forecasting": "future_state_forecasting",
        "segment_vs_subject": "future_state_forecasting",
        "motor_imagery": "motor_imagery_classification",
        "classification": "motor_imagery_classification",
        "motor_imery_classification": "motor_imagery_classification",
        "motor_imagery_classification": "motor_imagery_classification",
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported leakage-demo task: {task}")
    return aliases[normalized]


def _deprecated_task_alias_warning(task: str) -> str | None:
    normalized = str(task).strip().lower().replace("-", "_")
    if normalized == "motor_imery_classification":
        return "motor_imery_classification is deprecated; use motor_imagery_classification"
    return None


def _demo_dataset_label(summary: dict[str, Any], fallback: str) -> str:
    metadata = summary.get("metadata")
    if isinstance(metadata, dict):
        dataset = metadata.get("dataset")
        if dataset not in (None, ""):
            return str(dataset)
    datasets = summary.get("datasets")
    if isinstance(datasets, list) and len(datasets) == 1 and str(datasets[0]).strip():
        return str(datasets[0])
    return str(fallback)


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


def _aggregate_classification_seed_results(seed_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], list[float]] = {}
    for result in seed_results:
        for row in result.get("comparisons", []):
            if not isinstance(row, dict):
                continue
            split_id = str(row.get("split_id", "unknown"))
            model_id = str(row.get("model_id", "unknown"))
            for metric in (
                "accuracy",
                "balanced_accuracy",
                "f1",
                "auroc",
                "train_windows",
                "test_windows",
                "train_subjects",
                "test_subjects",
                "subject_overlap",
            ):
                value = row.get(metric)
                if isinstance(value, (int, float, np.floating)) and np.isfinite(float(value)):
                    by_key.setdefault((split_id, model_id, metric), []).append(float(value))
    aggregates = []
    for (split_id, model_id, metric), values in sorted(by_key.items()):
        aggregates.append({"split_id": split_id, "model_id": model_id, "metric": metric, **_numeric_summary(values)})
    return aggregates


def _aggregate_classification_inflation(seed_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_metric: dict[str, dict[str, list[float]]] = {}
    for result in seed_results:
        for row in result.get("inflation", []):
            if not isinstance(row, dict):
                continue
            metric = str(row.get("metric", "unknown"))
            for key in ("difference", "ratio"):
                value = row.get(key)
                if isinstance(value, (int, float, np.floating)) and np.isfinite(float(value)):
                    by_metric.setdefault(metric, {}).setdefault(key, []).append(float(value))
    rows = []
    for metric, values_by_key in sorted(by_metric.items()):
        row: dict[str, Any] = {"metric": metric}
        for key, values in sorted(values_by_key.items()):
            summary = _numeric_summary(values)
            row[key] = summary["mean"]
            row[f"{key}_ci_low"] = summary["ci_low"]
            row[f"{key}_ci_high"] = summary["ci_high"]
            row["n_seeds"] = summary["n_seeds"]
        rows.append(row)
    return rows


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


def _aggregate_classification_interpretation(seed_results: list[dict[str, Any]]) -> str:
    if not seed_results:
        return "failed"
    aggregate = _aggregate_classification_seed_results(seed_results)
    balanced_by_split = {
        row["split_id"]: row["mean"]
        for row in aggregate
        if row.get("metric") == "balanced_accuracy" and isinstance(row.get("mean"), (int, float))
    }
    bad_value = balanced_by_split.get("bad_segment_split")
    correct_value = balanced_by_split.get("correct_subject_heldout")
    if bad_value is None or correct_value is None:
        return "incomplete"
    if float(bad_value) > float(correct_value):
        return "bad_segment_split_inflates_classification_and_is_not_claim_eligible"
    return "bad_segment_split_did_not_inflate_classification_here_but_remains_not_claim_eligible"


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


def _classification_claim_gate(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    paper_gate = _paper_demo_gate(seed_results)
    return {
        "schema": "neurotwin.classification_leakage_claim_gate.v1",
        "passed": bool(paper_gate["passed"]),
        "scientific_claim_allowed": False,
        "bad_split_claim_allowed": False,
        "correct_split_claim_allowed": False,
        "bad_split_reason": "bad_segment_split is an intentional negative control with subject/window leakage",
        "valid_split_candidate": "correct_subject_heldout",
        "observed_seeds": paper_gate["observed_seeds"],
        "required_seeds": paper_gate["required_seeds"],
        "violations": paper_gate["violations"],
        "warnings": [
            *paper_gate["warnings"],
            "classification leakage diagnostics are never direct model-superiority claims",
        ],
    }


def _classification_figure_data(seed_results: list[dict[str, Any]], claim_gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "neurotwin.classification_leakage_figure.v1",
        "panels": {
            "split_counts": _classification_split_count_rows(seed_results),
            "classification_metric_comparison": _aggregate_classification_seed_results(seed_results),
            "identity_probe_reference": {
                "artifact": "identity_probe.json",
                "chance_metric": "chance_accuracy",
                "probe_metric": "accuracy",
            },
            "claim_gate": claim_gate,
        },
    }


def _classification_split_count_rows(seed_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in seed_results:
        seed = result.get("seed")
        for comparison in result.get("comparisons", []):
            if not isinstance(comparison, dict):
                continue
            rows.append(
                {
                    "seed": seed,
                    "split_id": comparison.get("split_id"),
                    "negative_control": comparison.get("negative_control"),
                    "train_windows": comparison.get("train_windows"),
                    "test_windows": comparison.get("test_windows"),
                    "train_subjects": comparison.get("train_subjects"),
                    "test_subjects": comparison.get("test_subjects"),
                    "subject_overlap": comparison.get("subject_overlap"),
                    "train_class_counts": comparison.get("train_class_counts", {}),
                    "test_class_counts": comparison.get("test_class_counts", {}),
                }
            )
    return rows


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


def _write_classification_demo_artifacts(out_dir: str | Path | None, payload: dict[str, Any]) -> None:
    if out_dir is None:
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_demo_payload(out, "classification_leakage_demo", payload)
    _write_demo_payload(out, "leakage_demo", payload)
    write_json(out / "classification_leakage_figure.json", payload.get("figure_data", {}))
    write_json(out / "classification_leakage_claim_gate.json", payload.get("claim_gate", {}))
    (out / "classification_leakage_demo.csv").write_text(_classification_demo_csv(payload), encoding="utf-8")
    (out / "split_comparison.csv").write_text(_classification_split_comparison_csv(payload), encoding="utf-8")
    (out / "classification_leakage_demo.md").write_text(_classification_demo_markdown(payload), encoding="utf-8")


def _classification_demo_csv(payload: dict[str, Any]) -> str:
    rows: list[tuple[Any, ...]] = []
    for result in payload.get("seed_results", []):
        if not isinstance(result, dict) or result.get("status") != "completed":
            continue
        seed = result.get("seed")
        for comparison in result.get("comparisons", []):
            if not isinstance(comparison, dict):
                continue
            for metric in ("accuracy", "balanced_accuracy", "f1", "auroc"):
                value = comparison.get(metric)
                if not isinstance(value, (int, float, np.floating)):
                    continue
                rows.append(
                    (
                        seed,
                        comparison.get("split_id", ""),
                        comparison.get("model_id", ""),
                        metric,
                        value,
                        comparison.get(f"{metric}_ci_low", ""),
                        comparison.get(f"{metric}_ci_high", ""),
                        comparison.get("train_windows", ""),
                        comparison.get("test_windows", ""),
                        comparison.get("subject_overlap", ""),
                        comparison.get("negative_control", ""),
                        comparison.get("scientific_claim_allowed", ""),
                    )
                )
    return _csv_rows(
        (
            "seed",
            "split_id",
            "model_id",
            "metric",
            "value",
            "ci_low",
            "ci_high",
            "train_windows",
            "test_windows",
            "subject_overlap",
            "negative_control",
            "scientific_claim_allowed",
        ),
        rows,
    )


def _classification_split_comparison_csv(payload: dict[str, Any]) -> str:
    rows: list[tuple[Any, ...]] = []
    for result in payload.get("seed_results", []):
        if not isinstance(result, dict) or result.get("status") != "completed":
            continue
        seed = result.get("seed")
        for comparison in result.get("comparisons", []):
            if not isinstance(comparison, dict):
                continue
            rows.append(
                (
                    seed,
                    comparison.get("split_id", ""),
                    comparison.get("negative_control", ""),
                    comparison.get("train_windows", ""),
                    comparison.get("test_windows", ""),
                    comparison.get("train_subjects", ""),
                    comparison.get("test_subjects", ""),
                    comparison.get("subject_overlap", ""),
                    comparison.get("class_count", ""),
                    comparison.get("status_detail", ""),
                )
            )
    return _csv_rows(
        (
            "seed",
            "split_id",
            "negative_control",
            "train_windows",
            "test_windows",
            "train_subjects",
            "test_subjects",
            "subject_overlap",
            "class_count",
            "status_detail",
        ),
        rows,
    )


def _classification_demo_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Motor-Imagery Classification Leakage Demo",
        "",
        f"- dataset: {payload.get('dataset')}",
        f"- task: {payload.get('task')}",
        f"- evidence_status: {payload.get('evidence_status')}",
        f"- scientific_claim_allowed: {payload.get('scientific_claim_allowed')}",
        f"- interpretation: {payload.get('interpretation')}",
        "",
        "## Claim Gate",
        "",
        f"- bad_split_claim_allowed: {_nested_bool(payload, 'claim_gate', 'bad_split_claim_allowed')}",
        f"- valid_split_candidate: {payload.get('claim_gate', {}).get('valid_split_candidate') if isinstance(payload.get('claim_gate'), dict) else 'unknown'}",
        "",
        "## Aggregate Metrics",
        "",
    ]
    for row in payload.get("seed_aggregate", []):
        if isinstance(row, dict) and row.get("metric") in {"accuracy", "balanced_accuracy", "f1", "auroc"}:
            lines.append(
                f"- {row.get('split_id')} {row.get('metric')}: "
                f"{row.get('mean')} [{row.get('ci_low')}, {row.get('ci_high')}]"
            )
    lines.extend(["", "## Limitations", "", "- Bad segment splits are intentional negative controls and are never claim eligible."])
    return "\n".join(lines) + "\n"


def _write_demo_payload(out_dir: str | Path | None, stem: str, payload: dict[str, Any]) -> None:
    if out_dir is None:
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"{stem}.json", payload)
    write_json(out / f"{stem.upper()}.json", payload)


def _csv_rows(header: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    lines = [",".join(header)]
    lines.extend(",".join(_csv_cell(value) for value in row) for row in rows)
    return "\n".join(lines) + "\n"


def _csv_cell(value: Any) -> str:
    text = str(value)
    if any(char in text for char in (",", "\"", "\n")):
        return "\"" + text.replace("\"", "\"\"") + "\""
    return text
