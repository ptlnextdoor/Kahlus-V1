from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from neurotwin.benchmarks.baseline_suite import EXECUTABLE_BASELINE_RUNNERS
from neurotwin.data.audit import audit_split_manifest
from neurotwin.data.prepared_tasks import SupervisedWindowTask
from neurotwin.eeg_v1.dataset import EEGV1Dataset, build_future_forecasting_task
from neurotwin.eeg_v1.gates import EEG_V1_CLAIM_SCOPE, build_eeg_v1_gate
from neurotwin.repro import write_json
from neurotwin.scoring.metrics import mae, mse, pearsonr, r2_score, rank_models

DEFAULT_EEG_V1_MODELS: tuple[str, ...] = (
    "persistence",
    "linear_ridge",
    "autoregressive_ridge",
    "mlp",
    "tcn",
    "transformer",
    "tiny_ssm",
    "neurotwin",
)

REQUIRED_EEG_V1_FIRST_CLASS_BASELINES: tuple[str, ...] = ("tiny_ssm",)
REQUIRED_EEG_V1_NEGATIVE_CONTROLS: tuple[str, ...] = ("shuffled_target_control",)
REQUIRES_EEG_V1_SHUFFLED_TARGET_DEGRADATION = True
REQUIRES_EEG_V1_SHUFFLED_TARGET_NOT_CLOSE_TO_REAL_BASELINES = True
EEG_V1_MIN_FORECAST_HORIZON = 1
EEG_V1_ALLOWED_SPLIT_TYPES: tuple[str, ...] = ("session_held_out", "subject_held_out")

REQUIRED_EEG_V1_CHECKSUM_ARTIFACTS: tuple[str, ...] = (
    "baseline_table.csv",
    "baseline_table.json",
    "baseline_verification.json",
    "dataset_summary.json",
    "diagnostic_report.md",
    "evidence_gate.json",
    "failure_reasons.json",
    "metrics.csv",
    "metrics.json",
    "per_channel_metrics.csv",
    "per_horizon_metrics.csv",
    "per_subject_metrics.csv",
    "run_config.json",
    "split_audit.json",
    "target_scale_context.json",
)

OPTIONAL_EEG_V1_CHECKSUM_ARTIFACTS: tuple[str, ...] = (
    "autocorrelation_diagnostics.csv",
    "autocorrelation_diagnostics.json",
)


def audit_eeg_v1_split(dataset: EEGV1Dataset, *, split_type: str = "subject_held_out") -> dict[str, Any]:
    policy = "subject" if split_type == "subject_held_out" else "session"
    report = audit_split_manifest(dataset.split_manifest, policy=policy)
    subjects = dataset.split_subjects
    subject_overlap = bool(
        (set(subjects["train"]) & set(subjects["val"]))
        or (set(subjects["train"]) & set(subjects["test"]))
        or (set(subjects["val"]) & set(subjects["test"]))
    )
    window_overlap = any("window overlap" in reason for reason in report.violations)
    failures = list(report.violations)
    if subject_overlap:
        failures.append("subject overlap across splits")
    return {
        "split_type": split_type,
        "train_subjects": list(subjects["train"]),
        "val_subjects": list(subjects["val"]),
        "test_subjects": list(subjects["test"]),
        "subject_overlap": subject_overlap,
        "window_overlap": window_overlap,
        "leakage_passed": bool(report.passed and not subject_overlap),
        "failure_reasons": failures,
        "checked": list(report.checked),
    }


def run_eeg_v1_baselines(
    task: SupervisedWindowTask,
    *,
    seed: int = 0,
    train_steps: int = 5,
    model_ids: Sequence[str] = DEFAULT_EEG_V1_MODELS,
) -> dict[str, Any]:
    predictions, failures = _predict_models(task, seed=seed, train_steps=train_steps, model_ids=tuple(model_ids))
    metrics_by_model = {model_id: _metric_bundle(task.y_test, pred) for model_id, pred in predictions.items()}
    ranking = [
        {"model_id": row.model_id, "metric": row.metric, "value": row.value, "rank": row.rank}
        for row in rank_models(metrics_by_model, metric="mse", higher_is_better=False)
    ]
    baseline_only = [row for row in ranking if row["model_id"] != "neurotwin"]
    best_model = str(baseline_only[0]["model_id"]) if baseline_only else None
    best_mse = float(baseline_only[0]["value"]) if baseline_only else None
    neuro_mse = _metric(metrics_by_model, "neurotwin", "mse")
    persistence_mse = _metric(metrics_by_model, "persistence", "mse")
    ridge_mse = _metric(metrics_by_model, "linear_ridge", "mse")
    result = {
        "task": task.task_id,
        "dataset": str(task.metadata.get("dataset_id", "unknown")),
        "source": str(task.metadata.get("source", "unknown")),
        "benchmark_status": str(task.metadata.get("benchmark_status", "unknown")),
        "seed": int(seed),
        "train_steps": int(train_steps),
        "metrics_by_model": metrics_by_model,
        "baseline_ranking": ranking,
        "failures": failures,
        "persistence_gap": _gap(neuro_mse, persistence_mse),
        "ridge_gap": _gap(neuro_mse, ridge_mse),
        "best_baseline": best_model,
        "best_baseline_mse": best_mse,
        "best_baseline_gap": _gap(neuro_mse, best_mse),
        "kahlus_beats_best_baseline": bool(neuro_mse is not None and best_mse is not None and neuro_mse < best_mse),
        "per_subject_metrics": _per_subject_metrics(task, predictions),
        "per_channel_metrics": _per_channel_metrics(task, predictions),
        "per_horizon_metrics": _per_horizon_metrics(task, predictions),
    }
    return result


def run_eeg_v1_autocorrelation_diagnostics(
    dataset: EEGV1Dataset,
    *,
    seed: int = 0,
    window_length: int = 8,
    forecast_horizon: int = 1,
    train_steps: int = 5,
    model_ids: Sequence[str] = ("persistence", "linear_ridge", "tiny_ssm"),
) -> dict[str, Any]:
    """Run controls that separate near-future autocorrelation from harder evidence."""

    models = tuple(model_ids)
    diagnostics: list[dict[str, Any]] = []
    short_task = build_future_forecasting_task(
        dataset,
        window_length=window_length,
        forecast_horizon=forecast_horizon,
        stride=1,
    )
    diagnostics.append(_task_diagnostic("short_horizon_overlap", short_task, seed=seed, train_steps=train_steps, model_ids=models))

    long_horizon = max(forecast_horizon * 4, forecast_horizon + window_length)
    diagnostics.append(
        _build_task_diagnostic(
            "long_horizon",
            dataset,
            seed=seed,
            train_steps=train_steps,
            model_ids=models,
            window_length=window_length,
            forecast_horizon=long_horizon,
            stride=1,
        )
    )
    diagnostics.append(
        _build_task_diagnostic(
            "non_overlapping_windows",
            dataset,
            seed=seed,
            train_steps=train_steps,
            model_ids=models,
            window_length=window_length,
            forecast_horizon=forecast_horizon,
            stride=window_length,
        )
    )
    diagnostics.append(
        _task_diagnostic(
            "shuffled_target_control",
            _shuffled_target_task(short_task, seed=seed + 1701),
            seed=seed,
            train_steps=train_steps,
            model_ids=_train_dependent_control_models(models),
        )
    )
    diagnostics.append(
        _task_diagnostic(
            "delta_prediction",
            _delta_target_task(short_task),
            seed=seed,
            train_steps=train_steps,
            model_ids=models,
        )
    )

    split = audit_eeg_v1_split(dataset, split_type="subject_held_out")
    diagnostics.append(
        {
            "diagnostic_id": "subject_held_out_split",
            "status": "completed",
            "leakage_passed": bool(split["leakage_passed"]),
            "subject_overlap": bool(split["subject_overlap"]),
            "window_overlap": bool(split["window_overlap"]),
            "failure_reasons": list(split["failure_reasons"]),
        }
    )
    diagnostics.append(_stimulus_task_split_diagnostic(dataset))
    return {"diagnostics": diagnostics, "summary": _autocorrelation_summary(diagnostics)}


def _build_task_diagnostic(
    diagnostic_id: str,
    dataset: EEGV1Dataset,
    *,
    seed: int,
    train_steps: int,
    model_ids: Sequence[str],
    window_length: int,
    forecast_horizon: int,
    stride: int,
) -> dict[str, Any]:
    try:
        task = build_future_forecasting_task(
            dataset,
            window_length=window_length,
            forecast_horizon=forecast_horizon,
            stride=stride,
        )
    except ValueError as exc:
        return {
            "diagnostic_id": diagnostic_id,
            "status": "skipped_insufficient_windows",
            "window_length": int(window_length),
            "forecast_horizon": int(forecast_horizon),
            "window_stride": int(stride),
            "reason": str(exc),
        }
    return _task_diagnostic(diagnostic_id, task, seed=seed, train_steps=train_steps, model_ids=model_ids)


def _task_diagnostic(
    diagnostic_id: str,
    task: SupervisedWindowTask,
    *,
    seed: int,
    train_steps: int,
    model_ids: Sequence[str],
) -> dict[str, Any]:
    result = run_eeg_v1_baselines(task, seed=seed, train_steps=train_steps, model_ids=model_ids)
    row = {
        "diagnostic_id": diagnostic_id,
        "status": "completed",
        "task_id": task.task_id,
        "window_length": int(task.metadata.get("window_length", 0)),
        "forecast_horizon": int(task.metadata.get("forecast_horizon", 0)),
        "window_stride": int(task.metadata.get("window_stride", 1)),
        "n_train_windows": int(task.x_train.shape[0]),
        "n_test_windows": int(task.x_test.shape[0]),
        "best_model": result["best_baseline"],
        "best_mse": result["best_baseline_mse"],
        "persistence_mse": _metric(result["metrics_by_model"], "persistence", "mse"),
        "linear_ridge_mse": _metric(result["metrics_by_model"], "linear_ridge", "mse"),
        "tiny_ssm_mse": _metric(result["metrics_by_model"], "tiny_ssm", "mse"),
        "metrics_by_model": result["metrics_by_model"],
        "failures": result["failures"],
    }
    if task.metadata.get("target_control") == "train_row_shuffled":
        row.update(
            {
                "shuffle_boundary": "train_split_only",
                "shuffle_seed": int(task.metadata["shuffle_seed"]),
                "shuffle_seed_source": "diagnostic_seed_plus_1701",
                "train_targets_shuffled": True,
                "validation_targets_shuffled": False,
                "test_targets_shuffled": False,
            }
        )
    return row


def _shuffled_target_task(task: SupervisedWindowTask, *, seed: int) -> SupervisedWindowTask:
    rng = np.random.default_rng(seed)
    return SupervisedWindowTask(
        task_id=f"{task.task_id}_shuffled_target_control",
        source_modality=task.source_modality,
        target_modality=task.target_modality,
        x_train=task.x_train,
        y_train=_permute_rows(task.y_train, rng),
        x_test=task.x_test,
        y_test=task.y_test,
        metric_mask=task.metric_mask,
        x_val=task.x_val,
        y_val=task.y_val,
        val_metric_mask=task.val_metric_mask,
        notes=tuple(task.notes) + (
            "Training target rows are shuffled to break input-target temporal alignment; validation and test targets are not shuffled.",
        ),
        metadata={
            **task.metadata,
            "target_control": "train_row_shuffled",
            "shuffle_boundary": "train_split_only",
            "shuffle_seed": int(seed),
        },
    )


def _train_dependent_control_models(model_ids: Sequence[str]) -> tuple[str, ...]:
    train_dependent = tuple(model_id for model_id in model_ids if model_id != "persistence")
    return train_dependent or tuple(model_ids)


def _delta_target_task(task: SupervisedWindowTask) -> SupervisedWindowTask:
    return SupervisedWindowTask(
        task_id=f"{task.task_id}_delta_prediction",
        source_modality=task.source_modality,
        target_modality=task.target_modality,
        x_train=task.x_train,
        y_train=task.y_train - task.x_train,
        x_test=task.x_test,
        y_test=task.y_test - task.x_test,
        metric_mask=task.metric_mask,
        x_val=task.x_val,
        y_val=(task.y_val - task.x_val) if task.x_val is not None and task.y_val is not None else None,
        val_metric_mask=task.val_metric_mask,
        notes=tuple(task.notes) + ("Target is future-minus-input delta, not the raw future signal.",),
        metadata={**task.metadata, "target_transform": "future_minus_input"},
    )


def _permute_rows(values: np.ndarray | None, rng: np.random.Generator) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.shape[0] < 2:
        return arr.copy()
    return arr[rng.permutation(arr.shape[0])].copy()


def _stimulus_task_split_diagnostic(dataset: EEGV1Dataset) -> dict[str, Any]:
    label_keys = {"stimulus_id", "stimulus", "task_id", "task", "condition", "event_type"}
    observed = sorted(_observed_label_keys(dataset, label_keys))
    if not observed:
        return {
            "diagnostic_id": "stimulus_task_held_out_split",
            "status": "not_applicable_missing_labels",
            "reason": "dataset has no stimulus/task label metadata to hold out",
        }
    labels_by_split = {
        split_name: _labels_for_records(records, observed)
        for split_name, records in (
            ("train", dataset.split_manifest.train),
            ("val", dataset.split_manifest.val),
            ("test", dataset.split_manifest.test),
        )
    }
    train_labels = labels_by_split["train"]
    val_labels = labels_by_split["val"]
    test_labels = labels_by_split["test"]
    overlaps = {
        "train_val": sorted(train_labels & val_labels),
        "train_test": sorted(train_labels & test_labels),
        "val_test": sorted(val_labels & test_labels),
    }
    failure_reasons = [
        f"{name}_label_overlap:{','.join(values)}"
        for name, values in overlaps.items()
        if values
    ]
    return {
        "diagnostic_id": "stimulus_task_held_out_split",
        "status": "completed",
        "observed_label_keys": observed,
        "leakage_passed": not failure_reasons,
        "label_overlap": bool(failure_reasons),
        "train_labels": sorted(train_labels),
        "val_labels": sorted(val_labels),
        "test_labels": sorted(test_labels),
        "failure_reasons": failure_reasons,
    }


def _observed_label_keys(dataset: EEGV1Dataset, label_keys: set[str]) -> set[str]:
    observed: set[str] = set()
    for record in dataset.split_manifest.all_records:
        if record.stimulus_id:
            observed.add("stimulus_id")
        observed.update(key for key in record.metadata if key in label_keys)
    for batch in dataset.batches:
        observed.update(key for key in getattr(batch, "metadata", {}) if key in label_keys)
    return observed


def _labels_for_records(records: Sequence[Any], label_keys: Sequence[str]) -> set[str]:
    labels: set[str] = set()
    for record in records:
        for key in label_keys:
            value = record.stimulus_id if key == "stimulus_id" and record.stimulus_id else record.metadata.get(key)
            if value is not None and str(value).strip():
                labels.add(str(value))
    return labels


def _autocorrelation_summary(diagnostics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(row["diagnostic_id"]): row for row in diagnostics}
    short = by_id.get("short_horizon_overlap", {})
    shuffled = by_id.get("shuffled_target_control", {})
    long = by_id.get("long_horizon", {})
    non_overlap = by_id.get("non_overlapping_windows", {})
    delta = by_id.get("delta_prediction", {})
    short_best = _optional_float(short.get("best_mse"))
    shuffled_best = _optional_float(shuffled.get("best_mse"))
    long_best = _optional_float(long.get("best_mse"))
    non_overlap_best = _optional_float(non_overlap.get("best_mse"))
    delta_best = _optional_float(delta.get("best_mse"))
    return {
        "autocorrelation_warning": (
            "Low short-horizon EEG MSE can be explained by temporal autocorrelation, "
            "overlapping windows, normalization scale, or synthetic fixture cleanliness."
        ),
        "short_horizon_best_mse": short_best,
        "shuffled_target_best_mse": shuffled_best,
        "shuffled_control_degrades": bool(shuffled_best is not None and short_best is not None and shuffled_best > short_best),
        "long_horizon_best_mse": long_best,
        "long_horizon_delta_vs_short": _subtract(long_best, short_best),
        "non_overlap_best_mse": non_overlap_best,
        "non_overlap_delta_vs_short": _subtract(non_overlap_best, short_best),
        "delta_prediction_best_mse": delta_best,
        "delta_prediction_delta_vs_short": _subtract(delta_best, short_best),
        "tiny_ssm_mse": _optional_float(short.get("tiny_ssm_mse")),
        "shuffled_target_control_mse": shuffled_best,
        "persistence_or_ridge_dominates": bool(short.get("best_model") in {"persistence", "linear_ridge", "autoregressive_ridge"}),
        "shuffled_target_close_to_real_baselines": _is_close(shuffled_best, short_best),
        "verdict": (
            "treat_v1_as_baseline_infrastructure_until_harder_controls_are_beaten"
            if short_best is not None
            else "insufficient_completed_diagnostics"
        ),
    }


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def _is_close(left: float | None, right: float | None) -> bool:
    if left is None or right is None or right <= 0:
        return False
    return bool(left <= right * 1.25)


def _predict_models(
    task: SupervisedWindowTask,
    *,
    seed: int,
    train_steps: int,
    model_ids: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    specs = {spec.model_id: spec for spec in EXECUTABLE_BASELINE_RUNNERS}
    predictions: dict[str, np.ndarray] = {}
    failures: list[dict[str, str]] = []
    for model_id in model_ids:
        spec = specs.get(model_id)
        if spec is None:
            failures.append({"model_id": model_id, "task_id": task.task_id, "reason": "unknown requested baseline model_id"})
            continue
        if not spec.supports(task):
            failures.append({"model_id": model_id, "task_id": task.task_id, "reason": "requested baseline is unavailable for task"})
            continue
        try:
            pred = np.asarray(spec.predict(task, seed, train_steps), dtype=np.float64)
            if pred.shape != task.y_test.shape:
                raise ValueError(f"prediction shape {pred.shape} does not match target {task.y_test.shape}")
            if not np.isfinite(pred).all():
                raise ValueError("prediction contains NaN or Inf")
            predictions[model_id] = pred
        except Exception as exc:  # noqa: BLE001 - baseline failures are artifact data.
            failures.append({"model_id": model_id, "task_id": task.task_id, "reason": str(exc)})
    return predictions, failures


def _metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mse": mse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "pearsonr": pearsonr(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def write_eeg_v1_artifacts(
    out_dir: str | Path,
    *,
    task: SupervisedWindowTask,
    baseline_result: dict[str, Any],
    split_audit: dict[str, Any],
    models: Sequence[str],
    train_steps: int,
    seed: int,
    autocorrelation_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    finite_metrics = _all_metrics_finite(baseline_result["metrics_by_model"])
    diagnostic_failures = _diagnostic_gate_failure_reasons(baseline_result, autocorrelation_diagnostics)
    model_win_claim = _model_win_claim_status(baseline_result, autocorrelation_diagnostics)
    result_payload = {**baseline_result, **model_win_claim}
    gate = build_eeg_v1_gate(
        dataset=str(task.metadata.get("dataset_id", baseline_result["dataset"])),
        split_audit_passed=bool(split_audit["leakage_passed"]),
        baseline_table_present=bool(baseline_result["metrics_by_model"]),
        finite_metrics=finite_metrics,
        forecast_horizon=int(task.metadata.get("forecast_horizon", 0)),
        split_type=str(split_audit["split_type"]),
        extra_failure_reasons=diagnostic_failures,
    )
    gate["gate_criteria"]["required_first_class_baselines"] = list(REQUIRED_EEG_V1_FIRST_CLASS_BASELINES)
    gate["gate_criteria"]["required_negative_controls"] = list(REQUIRED_EEG_V1_NEGATIVE_CONTROLS)
    gate["gate_criteria"]["requires_shuffled_target_degradation"] = REQUIRES_EEG_V1_SHUFFLED_TARGET_DEGRADATION
    gate["gate_criteria"]["requires_shuffled_target_not_close_to_real_baselines"] = (
        REQUIRES_EEG_V1_SHUFFLED_TARGET_NOT_CLOSE_TO_REAL_BASELINES
    )
    gate.update(model_win_claim)

    run_config = {
        "dataset": baseline_result["dataset"],
        "seed": int(seed),
        "train_steps": int(train_steps),
        "models": list(models),
        "window_length": int(task.metadata.get("window_length", 0)),
        "forecast_horizon": int(task.metadata.get("forecast_horizon", 0)),
        "sampling_rate_hz": task.metadata.get("sampling_rate_hz"),
        "data_source": str(task.metadata.get("source", "unknown")),
        "benchmark_status": str(task.metadata.get("benchmark_status", "unknown")),
        "selection_policy": "fixed_steps_symmetric",
        "claim_scope": EEG_V1_CLAIM_SCOPE,
    }
    dataset_summary = _dataset_summary(task, split_audit, result_payload)
    target_scale = _target_scale_context(task, result_payload)
    paths = {
        "metrics": write_json(out / "metrics.json", result_payload),
        "split_audit": write_json(out / "split_audit.json", split_audit),
        "evidence_gate": write_json(out / "evidence_gate.json", gate),
        "run_config": write_json(out / "run_config.json", run_config),
        "dataset_summary": write_json(out / "dataset_summary.json", dataset_summary),
        "target_scale_context": write_json(out / "target_scale_context.json", target_scale),
        "failure_reasons": write_json(
            out / "failure_reasons.json",
            {
                "baseline_failures": baseline_result["failures"],
                "gate_failures": gate["failure_reasons"],
                "split_audit_failures": list(split_audit["failure_reasons"]),
                "diagnostic_failures": diagnostic_failures,
            },
        ),
        "verification": write_json(out / "baseline_verification.json", _baseline_verification_payload(out_dir)),
    }
    paths["metrics_csv"] = _write_metrics_csv(out / "metrics.csv", baseline_result["metrics_by_model"])
    paths["baseline_table_json"] = write_json(
        out / "baseline_table.json",
        {"rows": _baseline_rows(baseline_result["metrics_by_model"]), "ranking": baseline_result["baseline_ranking"]},
    )
    paths["baseline_table_csv"] = _write_baseline_csv(out / "baseline_table.csv", baseline_result["metrics_by_model"])
    paths["per_subject_metrics"] = _write_rows(out / "per_subject_metrics.csv", baseline_result["per_subject_metrics"])
    paths["per_channel_metrics"] = _write_rows(out / "per_channel_metrics.csv", baseline_result["per_channel_metrics"])
    paths["per_horizon_metrics"] = _write_rows(out / "per_horizon_metrics.csv", baseline_result["per_horizon_metrics"])
    if autocorrelation_diagnostics is not None:
        paths["autocorrelation_diagnostics"] = write_json(
            out / "autocorrelation_diagnostics.json",
            autocorrelation_diagnostics,
        )
        paths["autocorrelation_diagnostics_csv"] = _write_autocorrelation_diagnostics_csv(
            out / "autocorrelation_diagnostics.csv",
            autocorrelation_diagnostics,
    )
    report_path = out / "diagnostic_report.md"
    report_path.write_text(
        _format_report(
            result_payload,
            split_audit,
            gate,
            run_config,
            autocorrelation_diagnostics,
            target_scale,
            dataset_summary,
        ),
        encoding="utf-8",
    )
    paths["diagnostic_report"] = report_path
    paths["checksum_manifest"] = _write_eeg_v1_checksum_manifest(out / "baseline_checksum_manifest.json", list(paths.values()))
    return paths


def _model_win_claim_status(
    baseline_result: dict[str, Any],
    autocorrelation_diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    best_baseline = str(baseline_result.get("best_baseline") or "unknown")
    kahlus_beats_best = bool(baseline_result.get("kahlus_beats_best_baseline"))
    summary = dict(autocorrelation_diagnostics.get("summary", {})) if autocorrelation_diagnostics else {}
    autocorrelation_baseline_won = best_baseline in {"persistence", "linear_ridge", "autoregressive_ridge"}
    if summary.get("persistence_or_ridge_dominates") is True:
        autocorrelation_baseline_won = True

    failure_reasons: list[str] = []
    if not kahlus_beats_best:
        failure_reasons.append("kahlus did not beat best baseline")
        if autocorrelation_baseline_won:
            failure_reasons.append(f"best baseline is {best_baseline}")
    if summary.get("shuffled_target_close_to_real_baselines") is True:
        failure_reasons.append("shuffled-target negative control is close to real baseline performance")
    if summary.get("shuffled_control_degrades") is False:
        failure_reasons.append("shuffled-target negative control did not degrade")

    if failure_reasons and autocorrelation_baseline_won:
        status = "blocked_by_autocorrelation_baseline"
    elif failure_reasons:
        status = "blocked_by_best_baseline"
    else:
        status = "allowed_model_beats_baselines"
    return {
        "model_win_claim_allowed": not failure_reasons,
        "model_win_status": status,
        "model_win_claim_failure_reasons": failure_reasons,
    }


def _diagnostic_gate_failure_reasons(
    baseline_result: dict[str, Any],
    autocorrelation_diagnostics: dict[str, Any] | None,
) -> list[str]:
    failures: list[str] = []
    completed_models = set(baseline_result.get("metrics_by_model", {}))
    for model_id in REQUIRED_EEG_V1_FIRST_CLASS_BASELINES:
        if model_id not in completed_models:
            failures.append(f"required first-class baseline missing: {model_id}")
    if autocorrelation_diagnostics is None:
        failures.extend(f"required negative control missing: {control}" for control in REQUIRED_EEG_V1_NEGATIVE_CONTROLS)
        return failures
    completed_diagnostics = {
        str(row.get("diagnostic_id"))
        for row in autocorrelation_diagnostics.get("diagnostics", [])
        if row.get("status") == "completed"
    }
    for control in REQUIRED_EEG_V1_NEGATIVE_CONTROLS:
        if control not in completed_diagnostics:
            failures.append(f"required negative control missing: {control}")
    summary = autocorrelation_diagnostics.get("summary", {})
    if (
        REQUIRES_EEG_V1_SHUFFLED_TARGET_DEGRADATION
        and summary.get("shuffled_control_degrades") is not True
    ):
        failures.append("shuffled-target negative control did not degrade")
    if (
        REQUIRES_EEG_V1_SHUFFLED_TARGET_NOT_CLOSE_TO_REAL_BASELINES
        and summary.get("shuffled_target_close_to_real_baselines") is not False
    ):
        failures.append("shuffled-target negative control is too close to real baseline performance")
    for row in autocorrelation_diagnostics.get("diagnostics", []):
        if row.get("diagnostic_id") != "stimulus_task_held_out_split":
            continue
        if row.get("status") != "completed" or bool(row.get("leakage_passed", True)):
            continue
        reasons = [str(reason) for reason in row.get("failure_reasons", []) if str(reason).strip()]
        detail = ", ".join(reasons) if reasons else "unknown_stimulus_task_label_overlap"
        failures.append(f"stimulus/task label split audit did not pass: {detail}")
    return failures


def _metric(metrics_by_model: dict[str, dict[str, float]], model_id: str, metric: str) -> float | None:
    value = metrics_by_model.get(model_id, {}).get(metric)
    return float(value) if value is not None and np.isfinite(float(value)) else None


def _gap(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return float(baseline - candidate)


def _dataset_summary(
    task: SupervisedWindowTask,
    split_audit: dict[str, Any],
    baseline_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": baseline_result["dataset"],
        "data_source": baseline_result["source"],
        "benchmark_status": baseline_result["benchmark_status"],
        "split_type": split_audit["split_type"],
        "n_train_subjects": len(split_audit.get("train_subjects", ())),
        "n_val_subjects": len(split_audit.get("val_subjects", ())),
        "n_test_subjects": len(split_audit.get("test_subjects", ())),
        "n_train_windows": int(np.asarray(task.x_train).shape[0]),
        "n_val_windows": int(0 if task.x_val is None else np.asarray(task.x_val).shape[0]),
        "n_test_windows": int(np.asarray(task.x_test).shape[0]),
        "window_length": int(task.metadata.get("window_length", np.asarray(task.x_train).shape[1])),
        "forecast_horizon": int(task.metadata.get("forecast_horizon", 0)),
        "window_stride": int(task.metadata.get("window_stride", 0)),
        "sampling_rate_hz": task.metadata.get("sampling_rate_hz"),
        "n_channels": int(np.asarray(task.x_train).shape[-1]),
    }


def _target_scale_context(task: SupervisedWindowTask, baseline_result: dict[str, Any]) -> dict[str, Any]:
    y_test = np.asarray(task.y_test, dtype=np.float64)
    target_std = float(np.std(y_test))
    target_variance = float(np.var(y_test))
    target_units = (
        "normalized_eeg_fixture_units"
        if task.metadata.get("source") == "synthetic_fixture"
        else "dataset_units_unknown_or_preprocessed"
    )
    return {
        "target_units": target_units,
        "scale_note": "MSE is reported in squared target units; compare it to target variance before interpreting magnitude.",
        "target_mean": float(np.mean(y_test)),
        "target_std": target_std,
        "target_variance": target_variance,
        "target_min": float(np.min(y_test)),
        "target_max": float(np.max(y_test)),
        "models": {
            model_id: _scale_context_for_model(metrics, target_std=target_std, target_variance=target_variance)
            for model_id, metrics in baseline_result["metrics_by_model"].items()
        },
    }


def _scale_context_for_model(metrics: dict[str, float], *, target_std: float, target_variance: float) -> dict[str, float | None]:
    model_mse = metrics.get("mse")
    if model_mse is None or not np.isfinite(float(model_mse)):
        return {"rmse": None, "mse_relative_to_target_variance": None, "rmse_relative_to_target_std": None}
    rmse = float(np.sqrt(float(model_mse)))
    return {
        "rmse": rmse,
        "mse_relative_to_target_variance": float(model_mse) / target_variance if target_variance > 0 else None,
        "rmse_relative_to_target_std": rmse / target_std if target_std > 0 else None,
    }


def _all_metrics_finite(metrics_by_model: dict[str, dict[str, float]]) -> bool:
    return bool(metrics_by_model) and all(np.isfinite(float(v)) for metrics in metrics_by_model.values() for v in metrics.values())


def _baseline_rows(metrics_by_model: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    return [
        {"model_id": model_id, "mse": values.get("mse"), "mae": values.get("mae"), "r2": values.get("r2"), "pearsonr": values.get("pearsonr"), "status": "completed"}
        for model_id, values in metrics_by_model.items()
    ]


def _method_order_rows(models: Sequence[str]) -> list[str]:
    return [
        f"| {idx} | {model_id} | {'main_model' if model_id == 'neurotwin' else 'baseline'} |"
        for idx, model_id in enumerate(models, start=1)
    ]


def _write_metrics_csv(path: Path, metrics_by_model: dict[str, dict[str, float]]) -> Path:
    rows = [
        {"model_id": model_id, "metric": metric, "value": value}
        for model_id, metrics in metrics_by_model.items()
        for metric, value in metrics.items()
    ]
    return _write_rows(path, rows, fieldnames=("model_id", "metric", "value"))


def _write_baseline_csv(path: Path, metrics_by_model: dict[str, dict[str, float]]) -> Path:
    return _write_rows(path, _baseline_rows(metrics_by_model), fieldnames=("model_id", "mse", "mae", "r2", "pearsonr", "status"))


def _write_autocorrelation_diagnostics_csv(path: Path, payload: dict[str, Any]) -> Path:
    rows = [
        {
            "diagnostic_id": row.get("diagnostic_id"),
            "status": row.get("status"),
            "window_length": row.get("window_length"),
            "forecast_horizon": row.get("forecast_horizon"),
            "window_stride": row.get("window_stride"),
            "n_train_windows": row.get("n_train_windows"),
            "n_test_windows": row.get("n_test_windows"),
            "best_model": row.get("best_model"),
            "best_mse": row.get("best_mse"),
            "persistence_mse": row.get("persistence_mse"),
            "linear_ridge_mse": row.get("linear_ridge_mse"),
            "tiny_ssm_mse": row.get("tiny_ssm_mse"),
            "leakage_passed": row.get("leakage_passed"),
            "reason": row.get("reason"),
        }
        for row in payload.get("diagnostics", [])
    ]
    return _write_rows(
        path,
        rows,
        fieldnames=(
            "diagnostic_id",
            "status",
            "window_length",
            "forecast_horizon",
            "window_stride",
            "n_train_windows",
            "n_test_windows",
            "best_model",
            "best_mse",
            "persistence_mse",
            "linear_ridge_mse",
            "tiny_ssm_mse",
            "leakage_passed",
            "reason",
        ),
    )


def _write_rows(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or (rows[0].keys() if rows else ("status",)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in names})
    return path


def _write_eeg_v1_checksum_manifest(path: Path, artifacts: Sequence[Path]) -> Path:
    rows = []
    for artifact in sorted(artifacts, key=lambda item: item.name):
        payload = artifact.read_bytes()
        rows.append(
            {
                "path": artifact.name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return write_json(
        path,
        {
            "schema": "kahlus.eeg_v1_baseline_checksums.v1",
            "algorithm": "sha256",
            "artifacts": rows,
        },
    )


def _baseline_verification_payload(out_dir: str | Path) -> dict[str, Any]:
    return {
        "schema": "kahlus.eeg_v1_baseline_verification.v1",
        "execution_lane": "local_cpu_or_single_process_only",
        "a100_jobs_launched": False,
        "checksum_manifest": "baseline_checksum_manifest.json",
        "checksum_audit_command": (
            "PYTHONPATH=src python3 scripts/audit_eeg_v1_baseline_checksums.py "
            f"--artifact-dir {out_dir}"
        ),
    }


def audit_eeg_v1_checksum_manifest(artifact_dir: str | Path) -> dict[str, Any]:
    root = Path(artifact_dir)
    manifest_path = root / "baseline_checksum_manifest.json"
    failure_reasons: list[str] = []
    if not manifest_path.exists():
        return {
            "schema": "kahlus.eeg_v1_baseline_checksum_audit.v1",
            "passed": False,
            "artifact_dir": str(root),
            "manifest": str(manifest_path),
            "artifacts_checked": 0,
            "failure_reasons": ["missing_manifest:baseline_checksum_manifest.json"],
        }

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "schema": "kahlus.eeg_v1_baseline_checksum_audit.v1",
            "passed": False,
            "artifact_dir": str(root),
            "manifest": str(manifest_path),
            "artifacts_checked": 0,
            "failure_reasons": ["invalid_manifest_json:baseline_checksum_manifest.json"],
        }

    if manifest.get("schema") != "kahlus.eeg_v1_baseline_checksums.v1":
        failure_reasons.append("unsupported_manifest_schema")
    if manifest.get("algorithm") != "sha256":
        failure_reasons.append("unsupported_checksum_algorithm")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        artifacts = []
        failure_reasons.append("invalid_artifacts_list")

    checked = 0
    manifest_paths: set[str] = set()
    allowed_manifest_paths = set(REQUIRED_EEG_V1_CHECKSUM_ARTIFACTS) | set(OPTIONAL_EEG_V1_CHECKSUM_ARTIFACTS)
    for row in artifacts:
        if not isinstance(row, dict):
            failure_reasons.append("invalid_artifact_row")
            continue
        rel_path = row.get("path")
        expected_bytes = row.get("bytes")
        expected_digest = row.get("sha256")
        if not isinstance(rel_path, str) or not rel_path or "/" in rel_path or "\\" in rel_path:
            failure_reasons.append("invalid_artifact_path")
            continue
        if rel_path in manifest_paths:
            failure_reasons.append(f"duplicate_manifest_entry:{rel_path}")
        manifest_paths.add(rel_path)
        if rel_path not in allowed_manifest_paths:
            failure_reasons.append(f"unexpected_manifest_entry:{rel_path}")
        artifact_path = root / rel_path
        if not artifact_path.exists():
            failure_reasons.append(f"missing_artifact:{rel_path}")
            continue
        payload = artifact_path.read_bytes()
        checked += 1
        if expected_bytes != len(payload):
            failure_reasons.append(f"bytes_mismatch:{rel_path}")
        if expected_digest != hashlib.sha256(payload).hexdigest():
            failure_reasons.append(f"checksum_mismatch:{rel_path}")

    for required_path in REQUIRED_EEG_V1_CHECKSUM_ARTIFACTS:
        if required_path not in manifest_paths:
            failure_reasons.append(f"missing_manifest_entry:{required_path}")
    _validate_gate_required_manifest_entries(root, manifest_paths, failure_reasons)
    _validate_baseline_verification_sidecar(root, failure_reasons)
    _validate_failure_reasons_sidecar(root, failure_reasons)
    _validate_split_audit_consistency(root, failure_reasons)
    _validate_model_win_claim_consistency(root, failure_reasons)
    _validate_autocorrelation_gate_consistency(root, failure_reasons)

    return {
        "schema": "kahlus.eeg_v1_baseline_checksum_audit.v1",
        "passed": not failure_reasons,
        "artifact_dir": str(root),
        "manifest": str(manifest_path),
        "artifacts_checked": checked,
        "failure_reasons": failure_reasons,
    }


def _validate_gate_required_manifest_entries(root: Path, manifest_paths: set[str], failure_reasons: list[str]) -> None:
    gate = _read_json_artifact(root / "evidence_gate.json", "evidence_gate", failure_reasons)
    if not isinstance(gate, dict) or gate.get("scientific_claim_allowed") is not True:
        return
    criteria = gate.get("gate_criteria")
    if not isinstance(criteria, dict) or not criteria.get("required_negative_controls"):
        return
    if "autocorrelation_diagnostics.json" not in manifest_paths:
        failure_reasons.append("gate_requires_manifest_entry:autocorrelation_diagnostics.json")


def _validate_baseline_verification_sidecar(root: Path, failure_reasons: list[str]) -> None:
    sidecar_path = root / "baseline_verification.json"
    if not sidecar_path.exists():
        failure_reasons.append("missing_artifact:baseline_verification.json")
        return
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        failure_reasons.append("invalid_verification_json")
        return
    if sidecar.get("schema") != "kahlus.eeg_v1_baseline_verification.v1":
        failure_reasons.append("invalid_verification_schema")
    if sidecar.get("execution_lane") != "local_cpu_or_single_process_only":
        failure_reasons.append("invalid_verification_execution_lane")
    if sidecar.get("a100_jobs_launched") is not False:
        failure_reasons.append("invalid_verification_a100_jobs_launched")
    if sidecar.get("checksum_manifest") != "baseline_checksum_manifest.json":
        failure_reasons.append("invalid_verification_checksum_manifest")
    expected_command = (
        "PYTHONPATH=src python3 scripts/audit_eeg_v1_baseline_checksums.py "
        f"--artifact-dir {root}"
    )
    if sidecar.get("checksum_audit_command") != expected_command:
        failure_reasons.append("invalid_verification_checksum_audit_command")


def _validate_failure_reasons_sidecar(root: Path, failure_reasons: list[str]) -> None:
    sidecar_path = root / "failure_reasons.json"
    if not sidecar_path.exists():
        failure_reasons.append("missing_artifact:failure_reasons.json")
        return
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        failure_reasons.append("invalid_failure_reasons_json")
        return
    if not isinstance(sidecar, dict):
        failure_reasons.append("invalid_failure_reasons_payload")
        return
    for field in ("baseline_failures", "gate_failures", "split_audit_failures", "diagnostic_failures"):
        if field not in sidecar:
            failure_reasons.append(f"missing_failure_reasons_field:{field}")
        elif not isinstance(sidecar[field], list):
            failure_reasons.append(f"invalid_failure_reasons_field:{field}")
    _validate_failure_reasons_baseline_consistency(root, sidecar, failure_reasons)
    _validate_failure_reasons_gate_consistency(root, sidecar, failure_reasons)
    _validate_failure_reasons_split_consistency(root, sidecar, failure_reasons)
    _validate_failure_reasons_diagnostic_consistency(root, sidecar, failure_reasons)


def _validate_failure_reasons_baseline_consistency(
    root: Path,
    failure_sidecar: dict[str, Any],
    failure_reasons: list[str],
) -> None:
    metrics = _read_json_artifact(root / "metrics.json", "metrics", failure_reasons)
    if not isinstance(metrics, dict) or not isinstance(metrics.get("failures"), list):
        failure_reasons.append("invalid_metrics_failures")
        return
    if failure_sidecar.get("baseline_failures") != metrics.get("failures"):
        failure_reasons.append("failure_reasons_baseline_failures_mismatch")


def _validate_failure_reasons_gate_consistency(
    root: Path,
    failure_sidecar: dict[str, Any],
    failure_reasons: list[str],
) -> None:
    gate_path = root / "evidence_gate.json"
    if not gate_path.exists():
        failure_reasons.append("missing_artifact:evidence_gate.json")
        return
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        failure_reasons.append("invalid_evidence_gate_json")
        return
    if not isinstance(gate, dict) or not isinstance(gate.get("failure_reasons"), list):
        failure_reasons.append("invalid_evidence_gate_failure_reasons")
        return
    if failure_sidecar.get("gate_failures") != gate.get("failure_reasons"):
        failure_reasons.append("failure_reasons_gate_failures_mismatch")


def _validate_failure_reasons_split_consistency(
    root: Path,
    failure_sidecar: dict[str, Any],
    failure_reasons: list[str],
) -> None:
    split_path = root / "split_audit.json"
    if not split_path.exists():
        failure_reasons.append("missing_artifact:split_audit.json")
        return
    try:
        split_audit = json.loads(split_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        failure_reasons.append("invalid_split_audit_json")
        return
    if not isinstance(split_audit, dict) or not isinstance(split_audit.get("failure_reasons"), list):
        failure_reasons.append("invalid_split_audit_failure_reasons")
        return
    if failure_sidecar.get("split_audit_failures") != split_audit.get("failure_reasons"):
        failure_reasons.append("failure_reasons_split_audit_failures_mismatch")


def _validate_failure_reasons_diagnostic_consistency(
    root: Path,
    failure_sidecar: dict[str, Any],
    failure_reasons: list[str],
) -> None:
    metrics = _read_json_artifact(root / "metrics.json", "metrics", failure_reasons)
    diagnostics_path = root / "autocorrelation_diagnostics.json"
    diagnostics = (
        _read_json_artifact(diagnostics_path, "autocorrelation_diagnostics", failure_reasons)
        if diagnostics_path.exists()
        else None
    )
    if not isinstance(metrics, dict):
        return
    expected = _diagnostic_gate_failure_reasons(metrics, diagnostics if isinstance(diagnostics, dict) else None)
    if failure_sidecar.get("diagnostic_failures") != expected:
        failure_reasons.append("failure_reasons_diagnostic_failures_mismatch")


def _validate_split_audit_consistency(root: Path, failure_reasons: list[str]) -> None:
    split_audit = _read_json_artifact(root / "split_audit.json", "split_audit", failure_reasons)
    if not isinstance(split_audit, dict):
        return
    train = _string_set(split_audit.get("train_subjects"))
    val = _string_set(split_audit.get("val_subjects"))
    test = _string_set(split_audit.get("test_subjects"))
    if train is None or val is None or test is None:
        failure_reasons.append("split_audit_invalid_subject_lists")
        return
    subject_overlap = bool((train & val) or (train & test) or (val & test))
    if split_audit.get("subject_overlap") is not subject_overlap:
        failure_reasons.append("split_audit_mismatch:subject_overlap")
    split_failures = split_audit.get("failure_reasons")
    if not isinstance(split_failures, list):
        failure_reasons.append("invalid_split_audit_failure_reasons")
        split_failures = []
    expected_leakage_passed = bool(not subject_overlap and not split_failures)
    if split_audit.get("leakage_passed") is not expected_leakage_passed:
        failure_reasons.append("split_audit_mismatch:leakage_passed")


def _string_set(values: Any) -> set[str] | None:
    if not isinstance(values, list):
        return None
    return {str(value) for value in values}


def _validate_model_win_claim_consistency(root: Path, failure_reasons: list[str]) -> None:
    metrics = _read_json_artifact(root / "metrics.json", "metrics", failure_reasons)
    gate = _read_json_artifact(root / "evidence_gate.json", "evidence_gate", failure_reasons)
    diagnostics_path = root / "autocorrelation_diagnostics.json"
    diagnostics = (
        _read_json_artifact(diagnostics_path, "autocorrelation_diagnostics", failure_reasons)
        if diagnostics_path.exists()
        else None
    )
    if not isinstance(metrics, dict) or not isinstance(gate, dict):
        return
    _validate_best_baseline_summary_consistency(metrics, failure_reasons)
    _validate_baseline_table_consistency(root, metrics, failure_reasons)
    _validate_metrics_csv_consistency(root, metrics, failure_reasons)
    _validate_granular_metrics_csv_consistency(root, metrics, failure_reasons)
    _validate_diagnostic_report_consistency(root, metrics, gate, diagnostics, failure_reasons)
    _validate_target_scale_context_consistency(root, metrics, failure_reasons)
    _validate_run_config_consistency(root, metrics, gate, failure_reasons)
    _validate_dataset_summary_consistency(root, metrics, failure_reasons)
    _validate_evidence_gate_metrics_consistency(metrics, gate, failure_reasons)
    _validate_evidence_gate_branch_consistency(gate, failure_reasons)
    _validate_evidence_gate_claim_scope_consistency(gate, failure_reasons)
    _validate_evidence_gate_dataset_consistency(metrics, gate, failure_reasons)
    _validate_evidence_gate_split_consistency(root, gate, failure_reasons)
    for field in ("model_win_claim_allowed", "model_win_status", "model_win_claim_failure_reasons"):
        if field not in metrics:
            failure_reasons.append(f"missing_model_win_field:metrics.json:{field}")
        if field not in gate:
            failure_reasons.append(f"missing_model_win_field:evidence_gate.json:{field}")
    for artifact_name, payload in (("metrics.json", metrics), ("evidence_gate.json", gate)):
        if "model_win_claim_allowed" in payload and not isinstance(payload.get("model_win_claim_allowed"), bool):
            failure_reasons.append(f"invalid_model_win_claim_allowed:{artifact_name}")
        if "model_win_status" in payload and not isinstance(payload.get("model_win_status"), str):
            failure_reasons.append(f"invalid_model_win_status:{artifact_name}")
        if "model_win_claim_failure_reasons" in payload and not isinstance(payload.get("model_win_claim_failure_reasons"), list):
            failure_reasons.append(f"invalid_model_win_claim_failure_reasons:{artifact_name}")
    for field in ("model_win_claim_allowed", "model_win_status", "model_win_claim_failure_reasons"):
        if field in metrics and field in gate and metrics.get(field) != gate.get(field):
            failure_reasons.append(f"model_win_metrics_gate_mismatch:{field}")
    recomputed = _model_win_claim_status(metrics, diagnostics if isinstance(diagnostics, dict) else None)
    for field, expected in recomputed.items():
        if metrics.get(field) != expected:
            failure_reasons.append(f"model_win_recomputed_mismatch:{field}")
        if gate.get(field) != expected:
            failure_reasons.append(f"model_win_gate_recomputed_mismatch:{field}")


def _validate_evidence_gate_metrics_consistency(
    metrics: dict[str, Any],
    gate: dict[str, Any],
    failure_reasons: list[str],
) -> None:
    if gate.get("baseline_table_present") is not _metrics_payload_present(metrics):
        failure_reasons.append("evidence_gate_mismatch:baseline_table_present")
    if gate.get("finite_metrics") is not _metrics_payload_finite(metrics):
        failure_reasons.append("evidence_gate_mismatch:finite_metrics")


def _validate_evidence_gate_branch_consistency(gate: dict[str, Any], failure_reasons: list[str]) -> None:
    if gate.get("branch") != "v1":
        failure_reasons.append("evidence_gate_mismatch:branch")


def _validate_evidence_gate_claim_scope_consistency(gate: dict[str, Any], failure_reasons: list[str]) -> None:
    if gate.get("claim_scope") != EEG_V1_CLAIM_SCOPE:
        failure_reasons.append("evidence_gate_mismatch:claim_scope")
    criteria = gate.get("gate_criteria")
    if not isinstance(criteria, dict) or criteria.get("allowed_claim_scope") != EEG_V1_CLAIM_SCOPE:
        failure_reasons.append("evidence_gate_mismatch:allowed_claim_scope")
    if not isinstance(criteria, dict) or criteria.get("min_forecast_horizon") != EEG_V1_MIN_FORECAST_HORIZON:
        failure_reasons.append("evidence_gate_mismatch:min_forecast_horizon")
    if (
        not isinstance(criteria, dict)
        or _criteria_tuple(criteria.get("allowed_split_types")) != EEG_V1_ALLOWED_SPLIT_TYPES
    ):
        failure_reasons.append("evidence_gate_mismatch:allowed_split_types")
    for field in (
        "requires_split_audit_passed",
        "requires_baseline_table_present",
        "requires_finite_metrics",
        "requires_calibration_checked",
    ):
        if not isinstance(criteria, dict) or criteria.get(field) is not True:
            failure_reasons.append(f"evidence_gate_mismatch:{field}")
    if (
        not isinstance(criteria, dict)
        or _criteria_tuple(criteria.get("required_first_class_baselines"))
        != REQUIRED_EEG_V1_FIRST_CLASS_BASELINES
    ):
        failure_reasons.append("evidence_gate_mismatch:required_first_class_baselines")
    if (
        not isinstance(criteria, dict)
        or _criteria_tuple(criteria.get("required_negative_controls")) != REQUIRED_EEG_V1_NEGATIVE_CONTROLS
    ):
        failure_reasons.append("evidence_gate_mismatch:required_negative_controls")
    if (
        not isinstance(criteria, dict)
        or criteria.get("requires_shuffled_target_degradation")
        is not REQUIRES_EEG_V1_SHUFFLED_TARGET_DEGRADATION
    ):
        failure_reasons.append("evidence_gate_mismatch:requires_shuffled_target_degradation")
    if (
        not isinstance(criteria, dict)
        or criteria.get("requires_shuffled_target_not_close_to_real_baselines")
        is not REQUIRES_EEG_V1_SHUFFLED_TARGET_NOT_CLOSE_TO_REAL_BASELINES
    ):
        failure_reasons.append("evidence_gate_mismatch:requires_shuffled_target_not_close_to_real_baselines")


def _criteria_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def _validate_evidence_gate_dataset_consistency(
    metrics: dict[str, Any],
    gate: dict[str, Any],
    failure_reasons: list[str],
) -> None:
    if gate.get("dataset") != metrics.get("dataset"):
        failure_reasons.append("evidence_gate_mismatch:dataset")


def _validate_evidence_gate_split_consistency(
    root: Path,
    gate: dict[str, Any],
    failure_reasons: list[str],
) -> None:
    split_audit = _read_json_artifact(root / "split_audit.json", "split_audit", failure_reasons)
    if not isinstance(split_audit, dict):
        return
    if gate.get("split_audit_passed") is not split_audit.get("leakage_passed"):
        failure_reasons.append("evidence_gate_mismatch:split_audit_passed")


def _metrics_payload_present(metrics: dict[str, Any]) -> bool:
    metrics_by_model = metrics.get("metrics_by_model")
    return isinstance(metrics_by_model, dict) and bool(metrics_by_model)


def _metrics_payload_finite(metrics: dict[str, Any]) -> bool:
    metrics_by_model = metrics.get("metrics_by_model")
    if not isinstance(metrics_by_model, dict) or not metrics_by_model:
        return False
    for values in metrics_by_model.values():
        if not isinstance(values, dict) or not values:
            return False
        for value in values.values():
            if _optional_float(value) is None:
                return False
    return True


def _validate_best_baseline_summary_consistency(metrics: dict[str, Any], failure_reasons: list[str]) -> None:
    if _normalize_ranking(metrics.get("baseline_ranking")) != _recompute_baseline_ranking(metrics):
        failure_reasons.append("baseline_ranking_recomputed_mismatch")
    recomputed = _recompute_best_baseline_summary(metrics)
    if not recomputed:
        failure_reasons.append("baseline_summary_recompute_failed")
        return
    for field, expected in recomputed.items():
        actual = metrics.get(field)
        if isinstance(expected, float):
            if not _float_values_match(_optional_float(actual), expected):
                failure_reasons.append(f"baseline_summary_recomputed_mismatch:{field}")
        elif actual != expected:
            failure_reasons.append(f"baseline_summary_recomputed_mismatch:{field}")


def _validate_baseline_table_consistency(root: Path, metrics: dict[str, Any], failure_reasons: list[str]) -> None:
    table = _read_json_artifact(root / "baseline_table.json", "baseline_table", failure_reasons)
    recomputed_rows = _normalize_baseline_rows(_recompute_baseline_rows(metrics))
    if isinstance(table, dict):
        if _normalize_baseline_rows(table.get("rows")) != recomputed_rows:
            failure_reasons.append("baseline_table_json_rows_recomputed_mismatch")
        if _normalize_ranking(table.get("ranking")) != _recompute_baseline_ranking(metrics):
            failure_reasons.append("baseline_table_json_ranking_recomputed_mismatch")
    csv_rows = _read_baseline_csv_rows(root / "baseline_table.csv", failure_reasons)
    if csv_rows is not None and _normalize_baseline_rows(csv_rows) != recomputed_rows:
            failure_reasons.append("baseline_table_csv_rows_recomputed_mismatch")


def _validate_diagnostic_report_consistency(
    root: Path,
    metrics: dict[str, Any],
    gate: dict[str, Any],
    diagnostics: dict[str, Any] | None,
    failure_reasons: list[str],
) -> None:
    report_path = root / "diagnostic_report.md"
    if not report_path.exists():
        failure_reasons.append("missing_artifact:diagnostic_report.md")
        return
    try:
        report = report_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        failure_reasons.append("invalid_diagnostic_report_text")
        return
    expected_lines = {
        "claim_scope": f"- claim_scope: {gate.get('claim_scope')}",
        "scientific_claim_allowed": f"- scientific_claim_allowed: {gate.get('scientific_claim_allowed')}",
        "best_baseline": f"- best_baseline: {metrics.get('best_baseline')}",
        "kahlus_beats_best_baseline": f"- kahlus_beats_best_baseline: {metrics.get('kahlus_beats_best_baseline')}",
        "model_win_claim_allowed": f"- model_win_claim_allowed: {gate.get('model_win_claim_allowed')}",
        "model_win_status": f"- model_win_status: {gate.get('model_win_status')}",
        "persistence_gap": f"- persistence_gap: {_report_value(metrics.get('persistence_gap'))}",
        "ridge_gap": f"- ridge_gap: {_report_value(metrics.get('ridge_gap'))}",
        "best_baseline_gap": f"- best_baseline_gap: {_report_value(metrics.get('best_baseline_gap'))}",
        "per_subject_rows": f"- per_subject_rows: {len(metrics.get('per_subject_metrics', ()))}",
        "per_channel_rows": f"- per_channel_rows: {len(metrics.get('per_channel_metrics', ()))}",
        "per_horizon_rows": f"- per_horizon_rows: {len(metrics.get('per_horizon_metrics', ()))}",
        "detailed_sidecars": (
            "- detailed_sidecars: per_subject_metrics.csv, per_channel_metrics.csv, per_horizon_metrics.csv"
        ),
    }
    expected_metric_breakdown_lines = [
        expected_lines["per_subject_rows"],
        expected_lines["per_channel_rows"],
        expected_lines["per_horizon_rows"],
        expected_lines["detailed_sidecars"],
    ]
    expected_baseline_gap_lines = [
        "- gap_definition: baseline_mse_minus_neurotwin_mse; positive means NeuroTwin/Kahlus lower MSE.",
        expected_lines["persistence_gap"],
        expected_lines["ridge_gap"],
        expected_lines["best_baseline_gap"],
        expected_lines["kahlus_beats_best_baseline"],
    ]
    expected_claim_boundary_lines = [
        "- This artifact supports only benchmark-readiness for EEG future-window forecasting.",
        "- It does not support diagnosis, treatment, epilepsy detection, depression detection, foundation-model, SOTA, v2, or v3 claims.",
    ]
    expected_hbn_boundary_lines = (
        ["This run used a user-provided HBN-style local manifest; not a public HBN benchmark result."]
        if metrics.get("benchmark_status") == "local_manifest_not_public_hbn_benchmark"
        else []
    )
    split_audit = _read_json_artifact(root / "split_audit.json", "split_audit", failure_reasons)
    expected_summary_header_lines: list[str] = []
    expected_split_failure_lines: list[str] = []
    if isinstance(split_audit, dict):
        expected_summary_header_lines = [
            "# Kahlus v1 EEG Baseline Diagnostic Report",
            f"- dataset: {metrics.get('dataset')}",
            f"- source: {metrics.get('source')}",
            f"- benchmark_status: {metrics.get('benchmark_status')}",
            f"- claim_scope: {gate.get('claim_scope')}",
            f"- scientific_claim_allowed: {gate.get('scientific_claim_allowed')}",
            f"- split_leakage_passed: {split_audit.get('leakage_passed')}",
            f"- best_baseline: {metrics.get('best_baseline')}",
            f"- kahlus_beats_best_baseline: {metrics.get('kahlus_beats_best_baseline')}",
        ]
        if isinstance(split_audit.get("failure_reasons"), list):
            expected_split_failure_lines = [f"- {reason}" for reason in split_audit["failure_reasons"]]
    expected_artifact_index_rows = _baseline_artifact_index_rows()
    for line in expected_artifact_index_rows:
        artifact = line.split(" |", 1)[0].removeprefix("| ")
        expected_lines[f"artifact_index:{artifact}"] = line
    dataset_summary = _read_json_artifact(root / "dataset_summary.json", "dataset_summary", failure_reasons)
    expected_dataset_summary_lines: list[str] = []
    if isinstance(dataset_summary, dict):
        dataset_summary_lines = {
            "dataset_summary_split_type": f"- split_type: {dataset_summary.get('split_type')}",
            "dataset_summary_n_train_subjects": f"- n_train_subjects: {dataset_summary.get('n_train_subjects')}",
            "dataset_summary_n_val_subjects": f"- n_val_subjects: {dataset_summary.get('n_val_subjects')}",
            "dataset_summary_n_test_subjects": f"- n_test_subjects: {dataset_summary.get('n_test_subjects')}",
            "dataset_summary_n_train_windows": f"- n_train_windows: {dataset_summary.get('n_train_windows')}",
            "dataset_summary_n_val_windows": f"- n_val_windows: {dataset_summary.get('n_val_windows')}",
            "dataset_summary_n_test_windows": f"- n_test_windows: {dataset_summary.get('n_test_windows')}",
            "dataset_summary_n_channels": f"- n_channels: {dataset_summary.get('n_channels')}",
            "dataset_summary_window_length": f"- window_length: {dataset_summary.get('window_length')}",
            "dataset_summary_forecast_horizon": f"- forecast_horizon: {dataset_summary.get('forecast_horizon')}",
            "dataset_summary_sampling_rate_hz": (
                f"- sampling_rate_hz: {_report_value(dataset_summary.get('sampling_rate_hz'))}"
            ),
        }
        expected_lines.update(dataset_summary_lines)
        expected_dataset_summary_lines = list(dataset_summary_lines.values())
    target_scale = _read_json_artifact(root / "target_scale_context.json", "target_scale_context", failure_reasons)
    expected_target_scale_rows: list[str] = []
    expected_target_scale_header_lines: list[str] = []
    if isinstance(target_scale, dict):
        target_scale_header_lines = {
            "target_units": f"- target_units: {target_scale.get('target_units')}",
            "target_std": f"- target_std: {_report_value(target_scale.get('target_std'))}",
            "target_variance": f"- target_variance: {_report_value(target_scale.get('target_variance'))}",
            "scale_note": f"- scale_note: {target_scale.get('scale_note')}",
        }
        expected_lines.update(target_scale_header_lines)
        expected_target_scale_header_lines = list(target_scale_header_lines.values())
        for model_id, values in target_scale.get("models", {}).items():
            if not isinstance(values, dict):
                continue
            expected_lines[f"target_scale_model:{model_id}"] = (
                "| "
                f"{model_id} | "
                f"{_report_value(values.get('rmse'))} | "
                f"{_report_value(values.get('rmse_relative_to_target_std'))} | "
                f"{_report_value(values.get('mse_relative_to_target_variance'))} |"
            )
            expected_target_scale_rows.append(expected_lines[f"target_scale_model:{model_id}"])
    verification = _read_json_artifact(root / "baseline_verification.json", "baseline_verification", failure_reasons)
    expected_checksum_audit_lines: list[str] = []
    if isinstance(verification, dict):
        expected_checksum_audit_lines = [
            f"- checksum_manifest: {verification.get('checksum_manifest')}",
            "- verification_sidecar: baseline_verification.json",
            f"- execution_lane: {verification.get('execution_lane')}",
            f"- a100_jobs_launched: {verification.get('a100_jobs_launched')}",
            "- command: `PYTHONPATH=src python3 scripts/audit_eeg_v1_baseline_checksums.py --artifact-dir <artifact-dir>`",
            "- `<artifact-dir>` is the directory containing this report and the listed baseline artifacts.",
        ]
        expected_lines.update(
            {
                "execution_lane": expected_checksum_audit_lines[2],
                "a100_jobs_launched": expected_checksum_audit_lines[3],
                "checksum_manifest": expected_checksum_audit_lines[0],
                "verification_sidecar": expected_checksum_audit_lines[1],
            }
        )
    run_config = _read_json_artifact(root / "run_config.json", "run_config", failure_reasons)
    expected_run_config_lines: dict[str, str] = {}
    expected_method_order_rows: list[str] = []
    expected_baseline_ranking_rows: list[str] = []
    expected_autocorrelation_intro_lines: list[str] = []
    expected_autocorrelation_summary_rows: list[str] = []
    expected_autocorrelation_dominance_lines: list[str] = []
    expected_autocorrelation_rows: list[str] = []
    expected_stimulus_task_audit_lines: list[str] = []
    if isinstance(run_config, dict):
        models = run_config.get("models")
        models_text = ", ".join(str(model_id) for model_id in models) if isinstance(models, list) else str(models)
        expected_run_config_lines.update(
            {
                "run_config_seed": f"- seed: {run_config.get('seed')}",
                "run_config_train_steps": f"- train_steps: {run_config.get('train_steps')}",
                "run_config_models": f"- models: {models_text}",
                "run_config_window_length": f"- window_length: {run_config.get('window_length')}",
                "run_config_forecast_horizon": f"- forecast_horizon: {run_config.get('forecast_horizon')}",
                "run_config_sampling_rate_hz": (
                    f"- sampling_rate_hz: {_report_value(run_config.get('sampling_rate_hz'))}"
                ),
                "run_config_data_source": f"- data_source: {run_config.get('data_source')}",
                "run_config_benchmark_status": f"- benchmark_status: {run_config.get('benchmark_status')}",
                "run_config_selection_policy": f"- selection_policy: {run_config.get('selection_policy')}",
                "run_config_claim_scope": f"- claim_scope: {run_config.get('claim_scope')}",
            }
        )
    if isinstance(run_config, dict) and isinstance(run_config.get("models"), list):
        expected_method_order_rows = _method_order_rows([str(model_id) for model_id in run_config["models"]])
        for row in expected_method_order_rows:
            parts = row.split(" | ")
            model_id = parts[1] if len(parts) > 1 else row
            expected_lines[f"method_order:{model_id}"] = row
    for row in _normalize_ranking(metrics.get("baseline_ranking")):
        model_id = str(row.get("model_id", ""))
        expected_lines[f"baseline_ranking:{model_id}"] = (
            f"| {row.get('rank')} | {model_id} | {_report_value(row.get('value'))} |"
        )
        expected_baseline_ranking_rows.append(expected_lines[f"baseline_ranking:{model_id}"])
    if isinstance(diagnostics, dict) and isinstance(diagnostics.get("summary"), dict):
        summary = diagnostics["summary"]
        expected_autocorrelation_intro_lines = [
            str(summary.get("autocorrelation_warning")),
            "- caveat: Low persistence or ridge MSE is autocorrelation evidence, not brain-state understanding.",
        ]
        expected_lines.update(
            {
                "autocorrelation_short_horizon_best_mse": (
                    f"| short_horizon_best_mse | {_report_value(summary.get('short_horizon_best_mse'))} |"
                ),
                "autocorrelation_persistence_or_ridge_dominates": (
                    "| persistence_or_ridge_dominates | "
                    f"{summary.get('persistence_or_ridge_dominates')} |"
                ),
                "autocorrelation_shuffled_target_close_to_real_baselines": (
                    "| shuffled_target_close_to_real_baselines | "
                    f"{summary.get('shuffled_target_close_to_real_baselines')} |"
                ),
                "autocorrelation_shuffled_control_degrades": (
                    f"| shuffled_control_degrades | {summary.get('shuffled_control_degrades')} |"
                ),
                "autocorrelation_long_horizon_delta_vs_short": (
                    f"| long_horizon_delta_vs_short | {_report_value(summary.get('long_horizon_delta_vs_short'))} |"
                ),
                "autocorrelation_non_overlap_delta_vs_short": (
                    f"| non_overlap_delta_vs_short | {_report_value(summary.get('non_overlap_delta_vs_short'))} |"
                ),
                "autocorrelation_delta_prediction_delta_vs_short": (
                    "| delta_prediction_delta_vs_short | "
                    f"{_report_value(summary.get('delta_prediction_delta_vs_short'))} |"
                ),
                "autocorrelation_verdict": f"| verdict | {summary.get('verdict')} |",
            }
        )
        expected_autocorrelation_summary_rows = [
            expected_lines["autocorrelation_short_horizon_best_mse"],
            f"| shuffled_target_best_mse | {_report_value(summary.get('shuffled_target_best_mse'))} |",
            f"| tiny_ssm_mse | {_report_value(summary.get('tiny_ssm_mse'))} |",
            f"| shuffled_target_control_mse | {_report_value(summary.get('shuffled_target_control_mse'))} |",
            expected_lines["autocorrelation_persistence_or_ridge_dominates"],
            expected_lines["autocorrelation_shuffled_target_close_to_real_baselines"],
            expected_lines["autocorrelation_shuffled_control_degrades"],
            expected_lines["autocorrelation_long_horizon_delta_vs_short"],
            expected_lines["autocorrelation_non_overlap_delta_vs_short"],
            expected_lines["autocorrelation_delta_prediction_delta_vs_short"],
            expected_lines["autocorrelation_verdict"],
        ]
        expected_autocorrelation_dominance_lines = [
            f"- persistence_or_ridge_dominates: {summary.get('persistence_or_ridge_dominates')}",
            f"- shuffled_target_close_to_real_baselines: {summary.get('shuffled_target_close_to_real_baselines')}",
        ]
        for row in diagnostics.get("diagnostics", []):
            if not isinstance(row, dict):
                continue
            best = row.get("best_mse")
            best_text = f"{float(best):.6g}" if best is not None else ""
            expected_lines[f"autocorrelation_row:{row.get('diagnostic_id')}"] = (
                f"| {row.get('diagnostic_id')} | {row.get('status')} | {row.get('best_model', '')} | "
                f"{best_text} | {_report_value(row.get('persistence_mse'))} | "
                f"{_report_value(row.get('linear_ridge_mse'))} | "
                f"{_report_value(row.get('tiny_ssm_mse'))} | {row.get('reason', '')} |"
            )
            expected_autocorrelation_rows.append(expected_lines[f"autocorrelation_row:{row.get('diagnostic_id')}"])
            if row.get("diagnostic_id") == "stimulus_task_held_out_split" and row.get(
                "status"
            ) != "not_applicable_missing_labels":
                expected_stimulus_task_audit_lines = [
                    f"- status: {row.get('status')}",
                    f"- leakage_passed: {row.get('leakage_passed')}",
                    f"- label_overlap: {row.get('label_overlap')}",
                    f"- observed_label_keys: {_report_list(row.get('observed_label_keys'))}",
                    f"- train_labels: {_report_list(row.get('train_labels'))}",
                    f"- val_labels: {_report_list(row.get('val_labels'))}",
                    f"- test_labels: {_report_list(row.get('test_labels'))}",
                    f"- failure_reasons: {_report_list(row.get('failure_reasons'))}",
                ]
    gate_criteria = gate.get("gate_criteria")
    expected_gate_criteria_lines: list[str] = []
    if isinstance(gate_criteria, dict):
        gate_criteria_lines = {
            "min_forecast_horizon": f"- min_forecast_horizon: {gate_criteria.get('min_forecast_horizon')}",
            "allowed_split_types": (
                "- allowed_split_types: "
                f"{', '.join(str(value) for value in gate_criteria.get('allowed_split_types', []))}"
            ),
            "requires_split_audit_passed": (
                f"- requires_split_audit_passed: {gate_criteria.get('requires_split_audit_passed')}"
            ),
            "requires_baseline_table_present": (
                f"- requires_baseline_table_present: {gate_criteria.get('requires_baseline_table_present')}"
            ),
            "requires_finite_metrics": f"- requires_finite_metrics: {gate_criteria.get('requires_finite_metrics')}",
            "requires_calibration_checked": (
                f"- requires_calibration_checked: {gate_criteria.get('requires_calibration_checked')}"
            ),
            "required_first_class_baselines": (
                "- requires_first_class_ssm_baseline: "
                f"{', '.join(str(value) for value in gate_criteria.get('required_first_class_baselines', []))}"
            ),
            "required_negative_controls": (
                "- requires_negative_control: "
                f"{', '.join(str(value) for value in gate_criteria.get('required_negative_controls', []))}"
            ),
            "requires_shuffled_target_degradation": (
                "- requires_shuffled_target_degradation: "
                f"{gate_criteria.get('requires_shuffled_target_degradation')}"
            ),
            "requires_shuffled_target_not_close_to_real_baselines": (
                "- requires_shuffled_target_not_close_to_real_baselines: "
                f"{gate_criteria.get('requires_shuffled_target_not_close_to_real_baselines')}"
            ),
            "allowed_claim_scope": f"- allowed_claim_scope: {gate_criteria.get('allowed_claim_scope')}",
        }
        expected_lines.update(gate_criteria_lines)
        expected_gate_criteria_lines = list(gate_criteria_lines.values())
    report_lines = report.splitlines()
    expected_gate_failure_lines = [f"- {reason}" for reason in gate.get("failure_reasons", [])]
    expected_baseline_failure_lines = [
        f"- {row.get('model_id', 'unknown')}: {row.get('reason', 'unknown failure')}"
        for row in metrics.get("failures", [])
        if isinstance(row, dict)
    ]
    singleton_fields = {"best_baseline", "persistence_gap", "ridge_gap", "best_baseline_gap"}
    for field, line in expected_lines.items():
        line_count = report_lines.count(line)
        if line_count == 0:
            failure_reasons.append(f"diagnostic_report_mismatch:{field}")
        elif field in singleton_fields and line_count > 1:
            failure_reasons.append(f"diagnostic_report_duplicate:{field}")
    if expected_run_config_lines:
        run_config_lines = _diagnostic_section_lines(report_lines, "## Run Config")
        for field, line in expected_run_config_lines.items():
            if line not in run_config_lines:
                failure_reasons.append(f"diagnostic_report_mismatch:{field}")
        if run_config_lines != list(expected_run_config_lines.values()):
            failure_reasons.append("diagnostic_report_mismatch:run_config_section")
    if (
        expected_dataset_summary_lines
        and _diagnostic_section_lines(report_lines, "## Dataset Summary") != expected_dataset_summary_lines
    ):
        failure_reasons.append("diagnostic_report_mismatch:dataset_summary_section")
    if _diagnostic_section_lines(report_lines, "## Metric Breakdown Summary") != expected_metric_breakdown_lines:
        failure_reasons.append("diagnostic_report_mismatch:metric_breakdown_section")
    if _diagnostic_section_lines(report_lines, "## Baseline Gap Summary") != expected_baseline_gap_lines:
        failure_reasons.append("diagnostic_report_mismatch:baseline_gap_section")
    if expected_gate_criteria_lines and _diagnostic_section_lines(report_lines, "## Evidence Gate Criteria") != expected_gate_criteria_lines:
        failure_reasons.append("diagnostic_report_mismatch:gate_criteria_section")
    for reason in gate.get("model_win_claim_failure_reasons", []):
        if f"- {reason}" not in report_lines:
            failure_reasons.append("diagnostic_report_mismatch:model_win_claim_failure_reason")
    expected_model_win_reasons = [f"- {reason}" for reason in gate.get("model_win_claim_failure_reasons", [])]
    if _model_win_reason_lines(report_lines) != expected_model_win_reasons:
        failure_reasons.append("diagnostic_report_mismatch:model_win_claim_failure_reasons")
    expected_model_win_section_lines = [
        expected_lines["model_win_claim_allowed"],
        expected_lines["model_win_status"],
        "- model_win_claim_failure_reasons:",
        *expected_model_win_reasons,
    ]
    if _diagnostic_section_lines(report_lines, "## Model Win Claim Status") != expected_model_win_section_lines:
        failure_reasons.append("diagnostic_report_mismatch:model_win_section")
    if expected_method_order_rows and _diagnostic_table_rows(report_lines, "| order | model_id | group |") != expected_method_order_rows:
        failure_reasons.append("diagnostic_report_mismatch:method_order_rows")
    if (
        expected_baseline_ranking_rows
        and _diagnostic_table_rows(report_lines, "| rank | model_id | mse |") != expected_baseline_ranking_rows
    ):
        failure_reasons.append("diagnostic_report_mismatch:baseline_ranking_rows")
    if _diagnostic_table_rows(report_lines, "| artifact | purpose |") != expected_artifact_index_rows:
        failure_reasons.append("diagnostic_report_mismatch:artifact_index_rows")
    if expected_checksum_audit_lines and _diagnostic_section_lines(report_lines, "## Checksum Audit") != expected_checksum_audit_lines:
        failure_reasons.append("diagnostic_report_mismatch:checksum_audit_section")
    if (
        expected_autocorrelation_intro_lines
        and _diagnostic_lines_until_heading(report_lines, "## Autocorrelation Diagnostics", "### ")
        != expected_autocorrelation_intro_lines
    ):
        failure_reasons.append("diagnostic_report_mismatch:autocorrelation_intro_lines")
    if (
        expected_autocorrelation_rows
        and _diagnostic_table_rows(
            report_lines,
            "| diagnostic | status | best_model | best_mse | persistence_mse | linear_ridge_mse | tiny_ssm_mse | reason |",
        )
        != expected_autocorrelation_rows
    ):
        failure_reasons.append("diagnostic_report_mismatch:autocorrelation_rows")
    if (
        expected_autocorrelation_summary_rows
        and _diagnostic_table_rows(report_lines, "| field | value |") != expected_autocorrelation_summary_rows
    ):
        failure_reasons.append("diagnostic_report_mismatch:autocorrelation_summary_rows")
    if (
        expected_autocorrelation_dominance_lines
        and _diagnostic_section_bullet_lines(report_lines, "### Baseline Dominance")
        != expected_autocorrelation_dominance_lines
    ):
        failure_reasons.append("diagnostic_report_mismatch:autocorrelation_dominance_lines")
    if (
        expected_target_scale_header_lines
        and _diagnostic_section_bullet_lines(report_lines, "## Target Scale Context") != expected_target_scale_header_lines
    ):
        failure_reasons.append("diagnostic_report_mismatch:target_scale_header_lines")
    if expected_target_scale_rows and sorted(_target_scale_model_lines(report_lines)) != sorted(expected_target_scale_rows):
        failure_reasons.append("diagnostic_report_mismatch:target_scale_model_rows")
    if _diagnostic_section_lines(report_lines, "## Claim Boundaries") != expected_claim_boundary_lines:
        failure_reasons.append("diagnostic_report_mismatch:claim_boundaries_section")
    if _diagnostic_section_lines(report_lines, "## HBN Local Path Boundary") != expected_hbn_boundary_lines:
        failure_reasons.append("diagnostic_report_mismatch:hbn_local_boundary_section")
    if expected_summary_header_lines and _diagnostic_preamble_lines(report_lines) != expected_summary_header_lines:
        failure_reasons.append("diagnostic_report_mismatch:summary_header_section")
    if _diagnostic_section_bullet_lines(report_lines, "## Gate Failures") != expected_gate_failure_lines:
        failure_reasons.append("diagnostic_report_mismatch:gate_failures_section")
    if _diagnostic_section_bullet_lines(report_lines, "## Split Audit Failures") != expected_split_failure_lines:
        failure_reasons.append("diagnostic_report_mismatch:split_audit_failures_section")
    if _diagnostic_section_bullet_lines(report_lines, "## Baseline Failures") != expected_baseline_failure_lines:
        failure_reasons.append("diagnostic_report_mismatch:baseline_failures_section")
    if _diagnostic_section_lines(report_lines, "## Stimulus/Task Split Audit") != expected_stimulus_task_audit_lines:
        failure_reasons.append("diagnostic_report_mismatch:stimulus_task_split_audit_section")


def _validate_run_config_consistency(
    root: Path,
    metrics: dict[str, Any],
    gate: dict[str, Any],
    failure_reasons: list[str],
) -> None:
    run_config = _read_json_artifact(root / "run_config.json", "run_config", failure_reasons)
    summary = _read_json_artifact(root / "dataset_summary.json", "dataset_summary", failure_reasons)
    if not isinstance(run_config, dict):
        return
    expected = {
        "dataset": metrics.get("dataset"),
        "data_source": metrics.get("source"),
        "benchmark_status": metrics.get("benchmark_status"),
        "claim_scope": gate.get("claim_scope"),
        "selection_policy": "fixed_steps_symmetric",
    }
    if isinstance(summary, dict):
        expected.update(
            {
                "window_length": summary.get("window_length"),
                "forecast_horizon": summary.get("forecast_horizon"),
                "sampling_rate_hz": summary.get("sampling_rate_hz"),
            }
        )
    for field, expected_value in expected.items():
        actual = run_config.get(field)
        if isinstance(expected_value, int):
            if _optional_int(actual) != expected_value:
                failure_reasons.append(f"run_config_mismatch:{field}")
        elif isinstance(expected_value, float):
            if not _float_values_match(_optional_float(actual), expected_value):
                failure_reasons.append(f"run_config_mismatch:{field}")
        elif actual != expected_value:
            failure_reasons.append(f"run_config_mismatch:{field}")
    metrics_by_model = metrics.get("metrics_by_model")
    models = run_config.get("models")
    if isinstance(metrics_by_model, dict) and isinstance(models, list):
        completed_models = {str(model_id) for model_id in metrics_by_model}
        configured_models = {str(model_id) for model_id in models}
        if not completed_models.issubset(configured_models):
            failure_reasons.append("run_config_mismatch:models")
    else:
        failure_reasons.append("run_config_mismatch:models")


def _validate_dataset_summary_consistency(root: Path, metrics: dict[str, Any], failure_reasons: list[str]) -> None:
    summary = _read_json_artifact(root / "dataset_summary.json", "dataset_summary", failure_reasons)
    split_audit = _read_json_artifact(root / "split_audit.json", "split_audit", failure_reasons)
    run_config = _read_json_artifact(root / "run_config.json", "run_config", failure_reasons)
    if not isinstance(summary, dict) or not isinstance(split_audit, dict) or not isinstance(run_config, dict):
        return
    expected = {
        "dataset": metrics.get("dataset"),
        "data_source": metrics.get("source"),
        "benchmark_status": metrics.get("benchmark_status"),
        "split_type": split_audit.get("split_type"),
        "n_train_subjects": len(split_audit.get("train_subjects", ())),
        "n_val_subjects": len(split_audit.get("val_subjects", ())),
        "n_test_subjects": len(split_audit.get("test_subjects", ())),
        "window_length": run_config.get("window_length"),
        "forecast_horizon": run_config.get("forecast_horizon"),
        "sampling_rate_hz": run_config.get("sampling_rate_hz"),
    }
    for field, expected_value in expected.items():
        actual = summary.get(field)
        if isinstance(expected_value, int):
            if _optional_int(actual) != expected_value:
                failure_reasons.append(f"dataset_summary_mismatch:{field}")
        elif isinstance(expected_value, float):
            if not _float_values_match(_optional_float(actual), expected_value):
                failure_reasons.append(f"dataset_summary_mismatch:{field}")
        elif actual != expected_value:
            failure_reasons.append(f"dataset_summary_mismatch:{field}")
    _validate_dataset_summary_granular_counts(root, summary, failure_reasons)


def _validate_dataset_summary_granular_counts(
    root: Path,
    summary: dict[str, Any],
    failure_reasons: list[str],
) -> None:
    expected_test_windows = _optional_int(summary.get("n_test_windows"))
    expected_channels = _optional_int(summary.get("n_channels"))
    if expected_test_windows is not None:
        rows = _read_metric_table_csv_rows(root / "per_subject_metrics.csv", failure_reasons)
        totals_by_model: dict[str, int] = {}
        if rows is not None:
            for row in rows:
                model_id = _optional_text(row.get("model_id"))
                n_windows = _optional_int(row.get("n_windows"))
                if model_id is None or n_windows is None:
                    failure_reasons.append("dataset_summary_mismatch:n_test_windows")
                    break
                totals_by_model[model_id] = totals_by_model.get(model_id, 0) + n_windows
            else:
                if not totals_by_model or any(total != expected_test_windows for total in totals_by_model.values()):
                    failure_reasons.append("dataset_summary_mismatch:n_test_windows")
    if expected_channels is not None:
        rows = _read_metric_table_csv_rows(root / "per_channel_metrics.csv", failure_reasons)
        channels_by_model: dict[str, set[int]] = {}
        if rows is not None:
            for row in rows:
                model_id = _optional_text(row.get("model_id"))
                channel = _optional_int(row.get("channel"))
                if model_id is None or channel is None:
                    failure_reasons.append("dataset_summary_mismatch:n_channels")
                    break
                channels_by_model.setdefault(model_id, set()).add(channel)
            else:
                if not channels_by_model or any(len(channels) != expected_channels for channels in channels_by_model.values()):
                    failure_reasons.append("dataset_summary_mismatch:n_channels")


def _model_win_reason_lines(report_lines: Sequence[str]) -> list[str]:
    try:
        start = report_lines.index("- model_win_claim_failure_reasons:") + 1
    except ValueError:
        return []
    rows: list[str] = []
    for line in report_lines[start:]:
        if not line:
            break
        if line.startswith("- "):
            rows.append(line)
    return rows


def _diagnostic_section_lines(report_lines: Sequence[str], heading: str) -> list[str]:
    try:
        start = report_lines.index(heading)
    except ValueError:
        return []
    rows: list[str] = []
    for line in report_lines[start + 1 :]:
        if line.startswith("## "):
            break
        if line:
            rows.append(line)
    return rows


def _diagnostic_preamble_lines(report_lines: Sequence[str]) -> list[str]:
    rows: list[str] = []
    for line in report_lines:
        if line.startswith("## "):
            break
        if line:
            rows.append(line)
    return rows


def _diagnostic_lines_until_heading(report_lines: Sequence[str], heading: str, stop_prefix: str) -> list[str]:
    try:
        start = report_lines.index(heading)
    except ValueError:
        return []
    rows: list[str] = []
    for line in report_lines[start + 1 :]:
        if line.startswith(stop_prefix):
            break
        if line:
            rows.append(line)
    return rows


def _diagnostic_table_rows(report_lines: Sequence[str], header: str) -> list[str]:
    try:
        start = report_lines.index(header) + 2
    except ValueError:
        return []
    rows: list[str] = []
    for line in report_lines[start:]:
        if not line:
            break
        if line.startswith("| "):
            rows.append(line)
    return rows


def _diagnostic_section_bullet_lines(report_lines: Sequence[str], heading: str) -> list[str]:
    try:
        start = report_lines.index(heading)
    except ValueError:
        return []
    rows: list[str] = []
    for line in report_lines[start + 1 :]:
        if line.startswith("## ") or line.startswith("| "):
            break
        if line.startswith("- "):
            rows.append(line)
    return rows


def _target_scale_model_lines(report_lines: Sequence[str]) -> list[str]:
    try:
        start = report_lines.index("| model_id | rmse | rmse_relative_to_target_std | mse_relative_to_target_variance |") + 2
    except ValueError:
        return []
    rows: list[str] = []
    for line in report_lines[start:]:
        if not line:
            break
        if line.startswith("| "):
            rows.append(line)
    return rows


def _validate_target_scale_context_consistency(
    root: Path,
    metrics: dict[str, Any],
    failure_reasons: list[str],
) -> None:
    target_scale = _read_json_artifact(root / "target_scale_context.json", "target_scale_context", failure_reasons)
    if not isinstance(target_scale, dict):
        return
    target_std = _optional_float(target_scale.get("target_std"))
    target_variance = _optional_float(target_scale.get("target_variance"))
    models = target_scale.get("models")
    metrics_by_model = metrics.get("metrics_by_model")
    if target_std is None or target_variance is None or not isinstance(models, dict) or not isinstance(metrics_by_model, dict):
        failure_reasons.append("target_scale_context_recompute_failed")
        return
    if not _float_values_match(target_variance, target_std * target_std):
        failure_reasons.append("target_scale_context_mismatch:target_variance")
    for model_id, model_metrics in metrics_by_model.items():
        if not isinstance(model_metrics, dict) or not isinstance(models.get(model_id), dict):
            failure_reasons.append(f"target_scale_context_missing_model:{model_id}")
            continue
        mse_value = _optional_float(model_metrics.get("mse"))
        expected = _scale_context_for_model(
            {"mse": mse_value} if mse_value is not None else {},
            target_std=target_std,
            target_variance=target_variance,
        )
        actual = models[model_id]
        for field, expected_value in expected.items():
            if not _float_values_match(_optional_float(actual.get(field)), expected_value):
                failure_reasons.append(f"target_scale_context_mismatch:{model_id}:{field}")


def _read_baseline_csv_rows(path: Path, failure_reasons: list[str]) -> list[dict[str, Any]] | None:
    if not path.exists():
        failure_reasons.append(f"missing_artifact:{path.name}")
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except csv.Error:
        failure_reasons.append("invalid_baseline_table_csv")
        return None


def _validate_metrics_csv_consistency(root: Path, metrics: dict[str, Any], failure_reasons: list[str]) -> None:
    csv_rows = _read_metrics_csv_rows(root / "metrics.csv", failure_reasons)
    if csv_rows is None:
        return
    if _normalize_metrics_csv_rows(csv_rows) != _normalize_metrics_csv_rows(_recompute_metrics_csv_rows(metrics)):
        failure_reasons.append("metrics_csv_recomputed_mismatch")


def _read_metrics_csv_rows(path: Path, failure_reasons: list[str]) -> list[dict[str, Any]] | None:
    if not path.exists():
        failure_reasons.append(f"missing_artifact:{path.name}")
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except csv.Error:
        failure_reasons.append("invalid_metrics_csv")
        return None


def _normalize_metrics_csv_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            return []
        value = _optional_float(row.get("value"))
        if value is None:
            return []
        normalized.append({"model_id": str(row.get("model_id")), "metric": str(row.get("metric")), "value": value})
    return sorted(normalized, key=lambda row: (row["model_id"], row["metric"]))


def _recompute_metrics_csv_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    metrics_by_model = metrics.get("metrics_by_model")
    if not isinstance(metrics_by_model, dict):
        return []
    return [
        {"model_id": str(model_id), "metric": str(metric), "value": value}
        for model_id, values in metrics_by_model.items()
        if isinstance(values, dict)
        for metric, value in values.items()
    ]


def _validate_granular_metrics_csv_consistency(root: Path, metrics: dict[str, Any], failure_reasons: list[str]) -> None:
    specs = (
        ("per_subject_metrics", "per_subject_metrics.csv", ("subject_id", "model_id"), ("n_windows",)),
        ("per_channel_metrics", "per_channel_metrics.csv", ("channel", "model_id"), ()),
        ("per_horizon_metrics", "per_horizon_metrics.csv", ("horizon_index", "model_id"), ()),
    )
    for metrics_field, filename, identity_fields, count_fields in specs:
        csv_rows = _read_metric_table_csv_rows(root / filename, failure_reasons)
        if csv_rows is None:
            continue
        if _normalize_metric_table_rows(csv_rows, identity_fields, count_fields) != _normalize_metric_table_rows(
            metrics.get(metrics_field),
            identity_fields,
            count_fields,
        ):
            failure_reasons.append(f"{filename.removesuffix('.csv')}_csv_recomputed_mismatch")


def _read_metric_table_csv_rows(path: Path, failure_reasons: list[str]) -> list[dict[str, Any]] | None:
    if not path.exists():
        failure_reasons.append(f"missing_artifact:{path.name}")
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except csv.Error:
        failure_reasons.append(f"invalid_{path.stem}_csv")
        return None


def _normalize_metric_table_rows(
    rows: Any,
    identity_fields: Sequence[str],
    count_fields: Sequence[str],
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    metric_fields = ("mse", "mae", "pearsonr", "r2")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            return []
        item: dict[str, Any] = {}
        for field in identity_fields:
            value = row.get(field)
            if field in {"channel", "horizon_index"}:
                value = _optional_int(value)
                if value is None:
                    return []
            else:
                value = str(value)
            item[field] = value
        for field in count_fields:
            value = _optional_int(row.get(field))
            if value is None:
                return []
            item[field] = value
        for field in metric_fields:
            value = _optional_float(row.get(field))
            if value is None:
                return []
            item[field] = value
        normalized.append(item)
    return sorted(normalized, key=lambda row: tuple(row[field] for field in (*identity_fields, *count_fields)))


def _optional_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_baseline_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            return []
        normalized.append(
            {
                "model_id": str(row.get("model_id")),
                "mse": _optional_float(row.get("mse")),
                "mae": _optional_float(row.get("mae")),
                "r2": _optional_float(row.get("r2")),
                "pearsonr": _optional_float(row.get("pearsonr")),
                "status": str(row.get("status")),
            }
        )
    return sorted(normalized, key=lambda row: row["model_id"])


def _recompute_baseline_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    metrics_by_model = metrics.get("metrics_by_model")
    if not isinstance(metrics_by_model, dict):
        return []
    return _baseline_rows(metrics_by_model)


def _normalize_ranking(ranking: Any) -> list[dict[str, Any]]:
    if not isinstance(ranking, list):
        return []
    rows: list[dict[str, Any]] = []
    for row in ranking:
        if not isinstance(row, dict):
            return []
        value = _optional_float(row.get("value"))
        if value is None:
            return []
        rows.append(
            {
                "model_id": str(row.get("model_id")),
                "metric": str(row.get("metric")),
                "value": value,
                "rank": int(row.get("rank")) if isinstance(row.get("rank"), int) else row.get("rank"),
            }
        )
    return rows


def _recompute_baseline_ranking(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    metrics_by_model = metrics.get("metrics_by_model")
    if not isinstance(metrics_by_model, dict):
        return []
    normalized_metrics: dict[str, dict[str, float]] = {}
    for model_id, values in metrics_by_model.items():
        if not isinstance(values, dict):
            continue
        mse_value = _optional_float(values.get("mse"))
        if mse_value is not None and np.isfinite(mse_value):
            normalized_metrics[str(model_id)] = {"mse": mse_value}
    return [
        {"model_id": row.model_id, "metric": row.metric, "value": float(row.value), "rank": row.rank}
        for row in rank_models(normalized_metrics, metric="mse", higher_is_better=False)
    ]


def _recompute_best_baseline_summary(metrics: dict[str, Any]) -> dict[str, Any] | None:
    metrics_by_model = metrics.get("metrics_by_model")
    if not isinstance(metrics_by_model, dict):
        return None
    baseline_mses: list[tuple[str, float]] = []
    for model_id, values in metrics_by_model.items():
        if model_id == "neurotwin" or not isinstance(values, dict):
            continue
        mse_value = _optional_float(values.get("mse"))
        if mse_value is not None and np.isfinite(mse_value):
            baseline_mses.append((str(model_id), mse_value))
    if not baseline_mses:
        return None
    best_baseline, best_mse = min(baseline_mses, key=lambda item: item[1])
    neuro_mse = _optional_float(metrics_by_model.get("neurotwin", {}).get("mse"))
    return {
        "best_baseline": best_baseline,
        "best_baseline_mse": best_mse,
        "best_baseline_gap": _gap(neuro_mse, best_mse),
        "kahlus_beats_best_baseline": bool(neuro_mse is not None and neuro_mse < best_mse),
    }


def _validate_autocorrelation_gate_consistency(root: Path, failure_reasons: list[str]) -> None:
    diagnostics_path = root / "autocorrelation_diagnostics.json"
    if not diagnostics_path.exists():
        return
    diagnostics = _read_json_artifact(diagnostics_path, "autocorrelation_diagnostics", failure_reasons)
    gate = _read_json_artifact(root / "evidence_gate.json", "evidence_gate", failure_reasons)
    run_config = _read_json_artifact(root / "run_config.json", "run_config", failure_reasons)
    if not isinstance(diagnostics, dict) or not isinstance(gate, dict):
        return
    _validate_autocorrelation_diagnostics_csv_consistency(root, diagnostics, failure_reasons)
    summary = diagnostics.get("summary")
    gate_failures = gate.get("failure_reasons")
    criteria = gate.get("gate_criteria")
    if gate.get("scientific_claim_allowed") is True and isinstance(criteria, dict):
        completed_rows = {
            str(row.get("diagnostic_id")): row
            for row in diagnostics.get("diagnostics", [])
            if isinstance(row, dict) and row.get("status") == "completed"
        }
        for control in criteria.get("required_negative_controls", []):
            control_id = str(control)
            if control_id not in completed_rows:
                failure_reasons.append(f"autocorrelation_gate_mismatch:required_negative_control:{control}")
            elif control_id == "shuffled_target_control":
                row = completed_rows[control_id]
                if (
                    row.get("shuffle_boundary") != "train_split_only"
                    or row.get("train_targets_shuffled") is not True
                    or row.get("validation_targets_shuffled") is not False
                    or row.get("test_targets_shuffled") is not False
                ):
                    failure_reasons.append(
                        "autocorrelation_gate_mismatch:shuffled_target_control_not_train_split_only"
                    )
                if not isinstance(row.get("shuffle_seed"), int) or not isinstance(
                    row.get("shuffle_seed_source"), str
                ):
                    failure_reasons.append(
                        "autocorrelation_gate_mismatch:shuffled_target_control_missing_seed_provenance"
                    )
                elif row.get("shuffle_seed_source") == "diagnostic_seed_plus_1701":
                    run_seed = run_config.get("seed") if isinstance(run_config, dict) else None
                    if not isinstance(run_seed, int) or row.get("shuffle_seed") != run_seed + 1701:
                        failure_reasons.append(
                            "autocorrelation_gate_mismatch:shuffled_target_control_seed_run_config_mismatch"
                        )
    if not isinstance(summary, dict) or not isinstance(gate_failures, list):
        return
    diagnostic_rows = {
        str(row.get("diagnostic_id")): row
        for row in diagnostics.get("diagnostics", [])
        if isinstance(row, dict) and row.get("status") == "completed"
    }
    for row in diagnostic_rows.values():
        _validate_autocorrelation_row_metric_fields(row, failure_reasons)
    _validate_autocorrelation_summary_mse_fields(summary, diagnostic_rows, failure_reasons)
    gate_failure_set = {str(reason) for reason in gate_failures}
    if (
        summary.get("shuffled_control_degrades") is False
        and "shuffled-target negative control did not degrade" not in gate_failure_set
    ):
        failure_reasons.append("autocorrelation_gate_mismatch:shuffled_control_degrades")
    if (
        summary.get("shuffled_target_close_to_real_baselines") is True
        and "shuffled-target negative control is too close to real baseline performance" not in gate_failure_set
    ):
        failure_reasons.append("autocorrelation_gate_mismatch:shuffled_target_close_to_real_baselines")


def _validate_autocorrelation_diagnostics_csv_consistency(
    root: Path,
    diagnostics: dict[str, Any],
    failure_reasons: list[str],
) -> None:
    csv_rows = _read_metric_table_csv_rows(root / "autocorrelation_diagnostics.csv", failure_reasons)
    if csv_rows is None:
        return
    if _normalize_autocorrelation_diagnostics_rows(csv_rows) != _normalize_autocorrelation_diagnostics_rows(
        diagnostics.get("diagnostics")
    ):
        failure_reasons.append("autocorrelation_diagnostics_csv_recomputed_mismatch")


def _validate_autocorrelation_row_metric_fields(row: dict[str, Any], failure_reasons: list[str]) -> None:
    diagnostic_id = str(row.get("diagnostic_id"))
    metrics_by_model = row.get("metrics_by_model")
    if not isinstance(metrics_by_model, dict):
        return
    for model_id, field in (
        ("persistence", "persistence_mse"),
        ("linear_ridge", "linear_ridge_mse"),
        ("tiny_ssm", "tiny_ssm_mse"),
    ):
        metrics = metrics_by_model.get(model_id)
        expected = _optional_float(metrics.get("mse")) if isinstance(metrics, dict) else None
        if not _float_values_match(_optional_float(row.get(field)), expected):
            failure_reasons.append(f"autocorrelation_gate_mismatch:diagnostic_metric:{diagnostic_id}:{field}")
    ranked = [
        (str(model_id), mse_value)
        for model_id, metrics in metrics_by_model.items()
        if isinstance(metrics, dict) and (mse_value := _optional_float(metrics.get("mse"))) is not None
    ]
    if not ranked:
        failure_reasons.append(f"autocorrelation_gate_mismatch:diagnostic_best_model:{diagnostic_id}")
        return
    expected_best_model, expected_best_mse = min(ranked, key=lambda item: item[1])
    if row.get("best_model") != expected_best_model:
        failure_reasons.append(f"autocorrelation_gate_mismatch:diagnostic_best_model:{diagnostic_id}")
    if not _float_values_match(_optional_float(row.get("best_mse")), expected_best_mse):
        failure_reasons.append(f"autocorrelation_gate_mismatch:diagnostic_best_mse:{diagnostic_id}")


def _normalize_autocorrelation_diagnostics_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    text_fields = ("diagnostic_id", "status", "best_model", "reason")
    int_fields = ("window_length", "forecast_horizon", "window_stride", "n_train_windows", "n_test_windows")
    float_fields = ("best_mse", "persistence_mse", "linear_ridge_mse", "tiny_ssm_mse")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            return []
        item: dict[str, Any] = {}
        for field in text_fields:
            item[field] = _optional_text(row.get(field))
        for field in int_fields:
            item[field] = _optional_int(row.get(field))
        for field in float_fields:
            item[field] = _optional_float(row.get(field))
        item["leakage_passed"] = _optional_bool(row.get("leakage_passed"))
        normalized.append(item)
    return sorted(normalized, key=lambda row: str(row["diagnostic_id"]))


def _validate_autocorrelation_summary_mse_fields(
    summary: dict[str, Any],
    diagnostic_rows: dict[str, dict[str, Any]],
    failure_reasons: list[str],
) -> None:
    short = diagnostic_rows.get("short_horizon_overlap", {})
    shuffled = diagnostic_rows.get("shuffled_target_control", {})
    long = diagnostic_rows.get("long_horizon", {})
    non_overlap = diagnostic_rows.get("non_overlapping_windows", {})
    delta = diagnostic_rows.get("delta_prediction", {})
    short_best = _optional_float(short.get("best_mse"))
    long_best = _optional_float(long.get("best_mse"))
    non_overlap_best = _optional_float(non_overlap.get("best_mse"))
    delta_best = _optional_float(delta.get("best_mse"))
    expected_fields = {
        "short_horizon_best_mse": short_best,
        "tiny_ssm_mse": _optional_float(short.get("tiny_ssm_mse")),
        "shuffled_target_best_mse": _optional_float(shuffled.get("best_mse")),
        "shuffled_target_control_mse": _optional_float(shuffled.get("best_mse")),
        "long_horizon_delta_vs_short": _subtract(long_best, short_best),
        "non_overlap_delta_vs_short": _subtract(non_overlap_best, short_best),
        "delta_prediction_delta_vs_short": _subtract(delta_best, short_best),
    }
    for field, expected in expected_fields.items():
        if not _float_values_match(_optional_float(summary.get(field)), expected):
            failure_reasons.append(f"autocorrelation_gate_mismatch:summary_{field}")
    expected_booleans = {
        "shuffled_control_degrades": bool(
            _optional_float(shuffled.get("best_mse")) is not None
            and short_best is not None
            and _optional_float(shuffled.get("best_mse")) > short_best
        ),
        "persistence_or_ridge_dominates": bool(
            short.get("best_model") in {"persistence", "linear_ridge", "autoregressive_ridge"}
        ),
        "shuffled_target_close_to_real_baselines": _is_close(_optional_float(shuffled.get("best_mse")), short_best),
    }
    for field, expected in expected_booleans.items():
        if summary.get(field) is not expected:
            failure_reasons.append(f"autocorrelation_gate_mismatch:summary_{field}")


def _float_values_match(left: float | None, right: float | None, *, tol: float = 1e-12) -> bool:
    if left is None or right is None:
        return left is right
    return abs(left - right) <= tol


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value == "True":
            return True
        if value == "False":
            return False
        if value == "":
            return None
    return None


def _read_json_artifact(path: Path, artifact_id: str, failure_reasons: list[str]) -> Any:
    if not path.exists():
        failure_reasons.append(f"missing_artifact:{path.name}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        failure_reasons.append(f"invalid_{artifact_id}_json")
        return None


def _per_subject_metrics(task: SupervisedWindowTask, predictions: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    subjects = list(task.metadata.get("test_subject_ids", ()))
    if not subjects:
        return []
    rows = []
    y_true = np.asarray(task.y_test)
    subject_arr = np.asarray(subjects)
    for model_id, pred in predictions.items():
        for subject in sorted(set(subjects)):
            mask = subject_arr == subject
            values = _metric_bundle(y_true[mask], pred[mask])
            rows.append(
                {
                    "subject_id": subject,
                    "model_id": model_id,
                    "mse": values["mse"],
                    "mae": values["mae"],
                    "pearsonr": values["pearsonr"],
                    "r2": values["r2"],
                    "n_windows": int(mask.sum()),
                }
            )
    return rows


def _per_channel_metrics(task: SupervisedWindowTask, predictions: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    y_true = np.asarray(task.y_test)
    for model_id, pred in predictions.items():
        for idx in range(y_true.shape[-1]):
            values = _metric_bundle(y_true[..., idx], pred[..., idx])
            rows.append({"channel": idx, "model_id": model_id, **values})
    return rows


def _per_horizon_metrics(task: SupervisedWindowTask, predictions: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    y_true = np.asarray(task.y_test)
    for model_id, pred in predictions.items():
        for idx in range(y_true.shape[1]):
            values = _metric_bundle(y_true[:, idx, :], pred[:, idx, :])
            rows.append({"horizon_index": idx, "model_id": model_id, **values})
    return rows


def _format_report(
    result: dict[str, Any],
    split_audit: dict[str, Any],
    gate: dict[str, Any],
    run_config: dict[str, Any],
    autocorrelation_diagnostics: dict[str, Any] | None = None,
    target_scale: dict[str, Any] | None = None,
    dataset_summary: dict[str, Any] | None = None,
) -> str:
    ranking = result["baseline_ranking"]
    gate_criteria = gate["gate_criteria"]
    lines = [
        "# Kahlus v1 EEG Baseline Diagnostic Report",
        "",
        f"- dataset: {result['dataset']}",
        f"- source: {result.get('source', 'unknown')}",
        f"- benchmark_status: {result.get('benchmark_status', 'unknown')}",
        f"- claim_scope: {gate['claim_scope']}",
        f"- scientific_claim_allowed: {gate['scientific_claim_allowed']}",
        f"- split_leakage_passed: {split_audit['leakage_passed']}",
        f"- best_baseline: {result['best_baseline']}",
        f"- kahlus_beats_best_baseline: {result['kahlus_beats_best_baseline']}",
        "",
        "## Artifact Index",
        "",
        "| artifact | purpose |",
        "| --- | --- |",
        *_baseline_artifact_index_rows(),
        "",
        "## Checksum Audit",
        "",
        "- checksum_manifest: baseline_checksum_manifest.json",
        "- verification_sidecar: baseline_verification.json",
        "- execution_lane: local_cpu_or_single_process_only",
        "- a100_jobs_launched: False",
        "- command: `PYTHONPATH=src python3 scripts/audit_eeg_v1_baseline_checksums.py --artifact-dir <artifact-dir>`",
        "- `<artifact-dir>` is the directory containing this report and the listed baseline artifacts.",
        "",
        "## Method Order",
        "",
        "| order | model_id | group |",
        "| --- | --- | --- |",
        *_method_order_rows(run_config["models"]),
        "",
        "## Baseline Ranking",
        "",
        "| rank | model_id | mse |",
        "| --- | --- | --- |",
    ]
    for row in ranking:
        lines.append(f"| {row['rank']} | {row['model_id']} | {float(row['value']):.6g} |")
    lines.extend(
        [
            "",
            "## Run Config",
            "",
            f"- seed: {run_config['seed']}",
            f"- train_steps: {run_config['train_steps']}",
            f"- models: {', '.join(run_config['models'])}",
            f"- window_length: {run_config['window_length']}",
            f"- forecast_horizon: {run_config['forecast_horizon']}",
            f"- sampling_rate_hz: {_report_value(run_config.get('sampling_rate_hz'))}",
            f"- data_source: {run_config['data_source']}",
            f"- benchmark_status: {run_config['benchmark_status']}",
            f"- selection_policy: {run_config['selection_policy']}",
            f"- claim_scope: {run_config['claim_scope']}",
        ]
    )
    if dataset_summary is not None:
        lines.extend(
            [
                "",
                "## Dataset Summary",
                "",
                f"- split_type: {dataset_summary.get('split_type')}",
                f"- n_train_subjects: {dataset_summary.get('n_train_subjects')}",
                f"- n_val_subjects: {dataset_summary.get('n_val_subjects')}",
                f"- n_test_subjects: {dataset_summary.get('n_test_subjects')}",
                f"- n_train_windows: {dataset_summary.get('n_train_windows')}",
                f"- n_val_windows: {dataset_summary.get('n_val_windows')}",
                f"- n_test_windows: {dataset_summary.get('n_test_windows')}",
                f"- n_channels: {dataset_summary.get('n_channels')}",
                f"- window_length: {dataset_summary.get('window_length')}",
                f"- forecast_horizon: {dataset_summary.get('forecast_horizon')}",
                f"- sampling_rate_hz: {_report_value(dataset_summary.get('sampling_rate_hz'))}",
            ]
        )
    if target_scale is not None:
        lines.extend(
            [
                "",
                "## Target Scale Context",
                "",
                f"- target_units: {target_scale['target_units']}",
                f"- target_std: {_report_value(target_scale.get('target_std'))}",
                f"- target_variance: {_report_value(target_scale.get('target_variance'))}",
                f"- scale_note: {target_scale['scale_note']}",
                "",
                "| model_id | rmse | rmse_relative_to_target_std | mse_relative_to_target_variance |",
                "| --- | --- | --- | --- |",
            ]
        )
        for model_id, values in target_scale.get("models", {}).items():
            lines.append(
                "| "
                f"{model_id} | "
                f"{_report_value(values.get('rmse'))} | "
                f"{_report_value(values.get('rmse_relative_to_target_std'))} | "
                f"{_report_value(values.get('mse_relative_to_target_variance'))} |"
            )
    lines.extend(
        [
            "",
            "## Baseline Gap Summary",
            "",
            "- gap_definition: baseline_mse_minus_neurotwin_mse; positive means NeuroTwin/Kahlus lower MSE.",
            f"- persistence_gap: {_report_value(result.get('persistence_gap'))}",
            f"- ridge_gap: {_report_value(result.get('ridge_gap'))}",
            f"- best_baseline_gap: {_report_value(result.get('best_baseline_gap'))}",
            f"- kahlus_beats_best_baseline: {result['kahlus_beats_best_baseline']}",
        ]
    )
    lines.extend(
        [
            "",
            "## Model Win Claim Status",
            "",
            f"- model_win_claim_allowed: {result.get('model_win_claim_allowed')}",
            f"- model_win_status: {result.get('model_win_status')}",
            "- model_win_claim_failure_reasons:",
            *[
                f"- {reason}"
                for reason in result.get("model_win_claim_failure_reasons", [])
            ],
        ]
    )
    per_subject_count = len(result.get("per_subject_metrics", ()))
    per_channel_count = len(result.get("per_channel_metrics", ()))
    per_horizon_count = len(result.get("per_horizon_metrics", ()))
    lines.extend(
        [
            "",
            "## Metric Breakdown Summary",
            "",
            f"- per_subject_rows: {per_subject_count}",
            f"- per_channel_rows: {per_channel_count}",
            f"- per_horizon_rows: {per_horizon_count}",
            "- detailed_sidecars: per_subject_metrics.csv, per_channel_metrics.csv, per_horizon_metrics.csv",
        ]
    )
    lines.extend(
        [
            "",
            "## Evidence Gate Criteria",
            "",
            f"- min_forecast_horizon: {gate_criteria['min_forecast_horizon']}",
            f"- allowed_split_types: {', '.join(gate_criteria['allowed_split_types'])}",
            f"- requires_split_audit_passed: {gate_criteria['requires_split_audit_passed']}",
            f"- requires_baseline_table_present: {gate_criteria['requires_baseline_table_present']}",
            f"- requires_finite_metrics: {gate_criteria['requires_finite_metrics']}",
            f"- requires_calibration_checked: {gate_criteria['requires_calibration_checked']}",
            f"- requires_first_class_ssm_baseline: {', '.join(gate_criteria.get('required_first_class_baselines', []))}",
            f"- requires_negative_control: {', '.join(gate_criteria.get('required_negative_controls', []))}",
            f"- requires_shuffled_target_degradation: {gate_criteria.get('requires_shuffled_target_degradation')}",
            "- requires_shuffled_target_not_close_to_real_baselines: "
            f"{gate_criteria.get('requires_shuffled_target_not_close_to_real_baselines')}",
            f"- allowed_claim_scope: {gate_criteria['allowed_claim_scope']}",
        ]
    )
    if result.get("benchmark_status") == "local_manifest_not_public_hbn_benchmark":
        lines.extend(
            [
                "",
                "## HBN Local Path Boundary",
                "",
                "This run used a user-provided HBN-style local manifest; not a public HBN benchmark result.",
            ]
        )
    if autocorrelation_diagnostics is not None:
        summary = autocorrelation_diagnostics["summary"]
        lines.extend(
            [
                "",
                "## Autocorrelation Diagnostics",
                "",
                summary["autocorrelation_warning"],
                "- caveat: Low persistence or ridge MSE is autocorrelation evidence, not brain-state understanding.",
                "",
                "### Summary",
                "",
                "| field | value |",
                "| --- | --- |",
                f"| short_horizon_best_mse | {_report_value(summary.get('short_horizon_best_mse'))} |",
                f"| shuffled_target_best_mse | {_report_value(summary.get('shuffled_target_best_mse'))} |",
                f"| tiny_ssm_mse | {_report_value(summary.get('tiny_ssm_mse'))} |",
                f"| shuffled_target_control_mse | {_report_value(summary.get('shuffled_target_control_mse'))} |",
                f"| persistence_or_ridge_dominates | {summary.get('persistence_or_ridge_dominates')} |",
                f"| shuffled_target_close_to_real_baselines | {summary.get('shuffled_target_close_to_real_baselines')} |",
                f"| shuffled_control_degrades | {summary.get('shuffled_control_degrades')} |",
                f"| long_horizon_delta_vs_short | {_report_value(summary.get('long_horizon_delta_vs_short'))} |",
                f"| non_overlap_delta_vs_short | {_report_value(summary.get('non_overlap_delta_vs_short'))} |",
                f"| delta_prediction_delta_vs_short | {_report_value(summary.get('delta_prediction_delta_vs_short'))} |",
                f"| verdict | {summary.get('verdict')} |",
                "",
                "",
                "### Baseline Dominance",
                "",
                "- persistence_or_ridge_dominates: "
                f"{summary.get('persistence_or_ridge_dominates')}",
                "- shuffled_target_close_to_real_baselines: "
                f"{summary.get('shuffled_target_close_to_real_baselines')}",
                "",
                "| diagnostic | status | best_model | best_mse | persistence_mse | linear_ridge_mse | tiny_ssm_mse | reason |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in autocorrelation_diagnostics.get("diagnostics", []):
            best = row.get("best_mse")
            best_text = f"{float(best):.6g}" if best is not None else ""
            lines.append(
                f"| {row.get('diagnostic_id')} | {row.get('status')} | {row.get('best_model', '')} | "
                f"{best_text} | {_report_value(row.get('persistence_mse'))} | "
                f"{_report_value(row.get('linear_ridge_mse'))} | "
                f"{_report_value(row.get('tiny_ssm_mse'))} | {row.get('reason', '')} |"
            )
        lines.extend(_stimulus_task_split_audit_report_lines(autocorrelation_diagnostics.get("diagnostics", [])))
    lines.extend(
        [
            "",
            "## Claim Boundaries",
            "",
            "- This artifact supports only benchmark-readiness for EEG future-window forecasting.",
            "- It does not support diagnosis, treatment, epilepsy detection, depression detection, foundation-model, SOTA, v2, or v3 claims.",
        ]
    )
    if gate["failure_reasons"]:
        lines.extend(["", "## Gate Failures", "", *[f"- {reason}" for reason in gate["failure_reasons"]]])
    if split_audit["failure_reasons"]:
        lines.extend(["", "## Split Audit Failures", "", *[f"- {reason}" for reason in split_audit["failure_reasons"]]])
    if result["failures"]:
        lines.extend(
            [
                "",
                "## Baseline Failures",
                "",
                *[f"- {row.get('model_id', 'unknown')}: {row.get('reason', 'unknown failure')}" for row in result["failures"]],
            ]
        )
    return "\n".join(lines) + "\n"


def _stimulus_task_split_audit_report_lines(diagnostics: Sequence[dict[str, Any]]) -> list[str]:
    audit = next((row for row in diagnostics if row.get("diagnostic_id") == "stimulus_task_held_out_split"), None)
    if audit is None or audit.get("status") == "not_applicable_missing_labels":
        return []
    return [
        "",
        "## Stimulus/Task Split Audit",
        "",
        f"- status: {audit.get('status')}",
        f"- leakage_passed: {audit.get('leakage_passed')}",
        f"- label_overlap: {audit.get('label_overlap')}",
        f"- observed_label_keys: {_report_list(audit.get('observed_label_keys'))}",
        f"- train_labels: {_report_list(audit.get('train_labels'))}",
        f"- val_labels: {_report_list(audit.get('val_labels'))}",
        f"- test_labels: {_report_list(audit.get('test_labels'))}",
        f"- failure_reasons: {_report_list(audit.get('failure_reasons'))}",
    ]


def _report_list(values: object) -> str:
    if values is None:
        return ""
    if isinstance(values, (list, tuple)):
        return ", ".join(str(value) for value in values)
    return str(values)


def _baseline_artifact_index_rows() -> list[str]:
    rows = [
        ("metrics.json", "aggregate metrics, ranking, gaps, and failure payload"),
        ("metrics.csv", "long-form metric table"),
        ("baseline_table.json", "baseline table and ranking sidecar"),
        ("baseline_table.csv", "baseline table in CSV form"),
        ("split_audit.json", "subject-held-out leakage audit"),
        ("evidence_gate.json", "benchmark-readiness gate decision and criteria"),
        ("run_config.json", "replay-critical run configuration"),
        ("dataset_summary.json", "bounded dataset and window geometry summary"),
        ("failure_reasons.json", "baseline, gate, and split-audit failure sidecar"),
        ("target_scale_context.json", "target-scale context for normalized MSE interpretation"),
        ("per_subject_metrics.csv", "subject-level metric breakdown"),
        ("per_channel_metrics.csv", "channel-level metric breakdown"),
        ("per_horizon_metrics.csv", "forecast-horizon metric breakdown"),
        ("autocorrelation_diagnostics.json", "optional autocorrelation control diagnostics"),
        ("autocorrelation_diagnostics.csv", "optional autocorrelation control diagnostics in CSV form"),
        ("baseline_verification.json", "local verification lane and checksum-audit command"),
        ("baseline_checksum_manifest.json", "SHA-256 manifest for the emitted evidence artifacts"),
    ]
    return [f"| {artifact} | {purpose} |" for artifact, purpose in rows]


def _report_value(value: object) -> str:
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.6g}" if np.isfinite(numeric) else ""
