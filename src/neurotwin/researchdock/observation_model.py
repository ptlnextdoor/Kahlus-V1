from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from neurotwin.repro import write_json
from neurotwin.researchdock.schemas import ResearchDockSession, ResearchDockTrial


@dataclass(frozen=True)
class ResearchDockObservationTask:
    task_id: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]
    test_subject_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ResearchDockObservationModelConfig:
    latent_dim: int = 2
    ridge_alpha: float = 1e-2
    subject_adapter_scale: float = 1.0


def build_researchdock_observation_task(
    sessions: Sequence[ResearchDockSession],
    *,
    seed: int = 0,
) -> ResearchDockObservationTask:
    if len(sessions) < 3:
        raise ValueError("ResearchDock observation task requires at least three sessions")
    ordered = tuple(sorted(sessions, key=lambda s: s.participant_id_hash))
    rng = np.random.default_rng(seed)
    indices = np.arange(len(ordered))
    rng.shuffle(indices)
    test_count = max(1, len(ordered) // 4)
    test_idx = _select_test_indices_with_targets(ordered, tuple(int(idx) for idx in indices), test_count)
    train_sessions = tuple(session for idx, session in enumerate(ordered) if idx not in test_idx)
    test_sessions = tuple(session for idx, session in enumerate(ordered) if idx in test_idx)
    x_train, y_train = _session_arrays(train_sessions)
    x_test, y_test = _session_arrays(test_sessions)
    if x_train.size == 0 or x_test.size == 0:
        raise ValueError("ResearchDock observation task requires nonempty train and test trials")
    missing_modality_audit = _missing_modality_audit(ordered)
    return ResearchDockObservationTask(
        task_id="researchdock_multimodal_observation",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        train_subjects=tuple(session.participant_id_hash for session in train_sessions),
        test_subjects=tuple(session.participant_id_hash for session in test_sessions),
        test_subject_ids=tuple(subject for session in test_sessions for subject in [session.participant_id_hash] * len(_eligible_trials(session))),
        feature_names=FEATURE_NAMES,
        target_names=TARGET_NAMES,
        metadata={
            "dataset_id": "researchdock_synthetic_v0",
            "split_type": "subject_held_out",
            "seed": int(seed),
            "claim_boundary": "synthetic_pretraining_only_no_clinical_claim",
            "missing_modality_audit": missing_modality_audit,
        },
    )


def run_researchdock_observation_benchmark(
    task: ResearchDockObservationTask,
    *,
    config: ResearchDockObservationModelConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ResearchDockObservationModelConfig()
    predictions = {
        "train_mean": _predict_train_mean(task.y_train, task.y_test.shape),
        "linear_ridge": _fit_ridge(task.x_train, task.y_train, task.x_test, alpha=cfg.ridge_alpha),
        "researchdock_observation_operator": _fit_observation_operator(task, cfg),
    }
    metrics_by_model = {model_id: _metrics(task.y_test, pred) for model_id, pred in predictions.items()}
    ranking = sorted(
        (
            {"model_id": model_id, "metric": "mse", "value": float(metrics["mse"]), "rank": 0}
            for model_id, metrics in metrics_by_model.items()
        ),
        key=lambda row: row["value"],
    )
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank
    baseline_rows = [row for row in ranking if row["model_id"] != "researchdock_observation_operator"]
    best_baseline = baseline_rows[0] if baseline_rows else None
    observation_mse = float(metrics_by_model["researchdock_observation_operator"]["mse"])
    return {
        "task_id": task.task_id,
        "dataset": task.metadata["dataset_id"],
        "model_order": ["train_mean", "linear_ridge", "researchdock_observation_operator"],
        "metrics_by_model": metrics_by_model,
        "baseline_ranking": ranking,
        "best_baseline": best_baseline["model_id"] if best_baseline else None,
        "best_baseline_mse": float(best_baseline["value"]) if best_baseline else None,
        "observation_operator_beats_best_baseline": bool(best_baseline is not None and observation_mse < float(best_baseline["value"])),
        "claim_boundary": "synthetic_pretraining_only_no_clinical_claim",
        "config": {
            "latent_dim": int(cfg.latent_dim),
            "ridge_alpha": float(cfg.ridge_alpha),
            "subject_adapter_scale": float(cfg.subject_adapter_scale),
        },
        "missing_modality_audit": task.metadata.get("missing_modality_audit", {}),
        "split": {
            "train_subjects": list(task.train_subjects),
            "test_subjects": list(task.test_subjects),
            "subject_overlap": bool(set(task.train_subjects) & set(task.test_subjects)),
        },
    }


def write_researchdock_observation_artifacts(
    out_dir: str | Path,
    *,
    task: ResearchDockObservationTask,
    result: dict[str, Any],
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": write_json(out / "researchdock_observation_metrics.json", result),
        "task": write_json(
            out / "researchdock_observation_task.json",
            {
                "task_id": task.task_id,
                "feature_names": list(task.feature_names),
                "target_names": list(task.target_names),
                "train_subjects": list(task.train_subjects),
                "test_subjects": list(task.test_subjects),
                "metadata": task.metadata,
            },
        ),
    }
    paths["split_audit"] = write_json(
        out / "researchdock_observation_split_audit.json",
        _observation_split_audit(result),
    )
    paths["baseline_table_csv"] = _write_baseline_csv(out / "researchdock_observation_baselines.csv", result)
    report = out / "researchdock_observation_report.md"
    report.write_text(_format_observation_report(result), encoding="utf-8")
    paths["report"] = report
    return paths


FEATURE_NAMES: tuple[str, ...] = (
    "task_reward_anticipation",
    "task_effort_for_reward",
    "task_recovery_rest",
    "reward_condition",
    "effort_level",
    "reaction_time_ms",
    "accuracy",
    "self_report_valence",
    "self_report_arousal",
    "self_report_motivation",
)
TARGET_NAMES: tuple[str, ...] = ("pupil_diameter", "hrv_proxy")


def _observation_split_audit(result: dict[str, Any]) -> dict[str, Any]:
    split = result.get("split", {})
    train_subjects = list(split.get("train_subjects", ()))
    test_subjects = list(split.get("test_subjects", ()))
    subject_overlap = bool(split.get("subject_overlap"))
    failures: list[str] = []
    if not train_subjects:
        failures.append("missing train subjects")
    if not test_subjects:
        failures.append("missing test subjects")
    if subject_overlap:
        failures.append("subject overlap across train/test split")
    return {
        "split_type": "subject_held_out",
        "n_train_subjects": len(train_subjects),
        "n_test_subjects": len(test_subjects),
        "subject_overlap": subject_overlap,
        "leakage_passed": not failures,
        "failure_reasons": failures,
    }


def _select_test_indices_with_targets(
    sessions: Sequence[ResearchDockSession],
    shuffled_indices: tuple[int, ...],
    test_count: int,
) -> set[int]:
    selected = set(shuffled_indices[:test_count])
    if any(_eligible_trials(sessions[idx]) for idx in selected):
        return selected
    for idx in shuffled_indices[test_count:]:
        if _eligible_trials(sessions[idx]):
            return {int(idx)}
    return selected


def _session_arrays(sessions: Sequence[ResearchDockSession]) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[list[float]] = []
    y_rows: list[list[float]] = []
    for session in sessions:
        for trial in _eligible_trials(session):
            x_rows.append(_trial_features(trial))
            y_rows.append(_trial_targets(trial))
    if not x_rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64), np.empty((0, len(TARGET_NAMES)), dtype=np.float64)
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    return x, y


def _eligible_trials(session: ResearchDockSession) -> tuple[ResearchDockTrial, ...]:
    return tuple(
        trial
        for trial in session.trials
        if trial.sensor_packet is not None
        and trial.sensor_packet.pupil_diameter is not None
        and trial.sensor_packet.hrv_proxy is not None
        and trial.reaction_time_ms is not None
        and trial.accuracy is not None
    )


def _missing_modality_audit(sessions: Sequence[ResearchDockSession]) -> dict[str, Any]:
    counts = {
        "missing_sensor_packet": 0,
        "missing_pupil_diameter": 0,
        "missing_hrv_proxy": 0,
        "missing_behavior_response": 0,
    }
    total = 0
    eligible = 0
    for session in sessions:
        for trial in session.trials:
            total += 1
            reasons = _trial_missing_modality_reasons(trial)
            if reasons:
                for reason in reasons:
                    counts[reason] += 1
            else:
                eligible += 1
    return {
        "total_trials": total,
        "eligible_trials": eligible,
        "skipped_trials": total - eligible,
        **counts,
        "skip_reasons": [key for key, value in counts.items() if value > 0],
    }


def _trial_missing_modality_reasons(trial: ResearchDockTrial) -> tuple[str, ...]:
    reasons: list[str] = []
    if trial.sensor_packet is None:
        reasons.append("missing_sensor_packet")
        reasons.append("missing_pupil_diameter")
        reasons.append("missing_hrv_proxy")
    else:
        if trial.sensor_packet.pupil_diameter is None:
            reasons.append("missing_pupil_diameter")
        if trial.sensor_packet.hrv_proxy is None:
            reasons.append("missing_hrv_proxy")
    if trial.reaction_time_ms is None or trial.accuracy is None:
        reasons.append("missing_behavior_response")
    return tuple(reasons)


def _trial_features(trial: ResearchDockTrial) -> list[float]:
    report = trial.self_report
    return [
        1.0 if trial.task_name == "reward_anticipation" else 0.0,
        1.0 if trial.task_name == "effort_for_reward" else 0.0,
        1.0 if trial.task_name == "recovery_rest" else 0.0,
        1.0 if trial.reward_condition == "reward" else 0.0,
        _value(trial.effort_level),
        _value(trial.reaction_time_ms) / 1000.0,
        _value(trial.accuracy),
        _value(report.valence if report is not None else None),
        _value(report.arousal if report is not None else None),
        _value(report.motivation if report is not None else None),
    ]


def _trial_targets(trial: ResearchDockTrial) -> list[float]:
    assert trial.sensor_packet is not None
    return [_value(trial.sensor_packet.pupil_diameter), _value(trial.sensor_packet.hrv_proxy) / 100.0]


def _value(value: float | None) -> float:
    if value is None:
        return 0.0
    out = float(value)
    return out if np.isfinite(out) else 0.0


def _predict_train_mean(y_train: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    mean = np.mean(y_train, axis=0, keepdims=True)
    return np.broadcast_to(mean, target_shape).astype(np.float64).copy()


def _fit_ridge(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, *, alpha: float) -> np.ndarray:
    x_aug = _augment(x_train)
    penalty = np.sqrt(max(float(alpha), 0.0)) * np.eye(x_aug.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    design = np.vstack([x_aug, penalty])
    target = np.vstack([y_train, np.zeros((penalty.shape[0], y_train.shape[1]), dtype=np.float64)])
    weights, *_ = np.linalg.lstsq(design, target, rcond=None)
    return _augment(x_test) @ weights


def _fit_observation_operator(task: ResearchDockObservationTask, cfg: ResearchDockObservationModelConfig) -> np.ndarray:
    ridge = _fit_ridge(task.x_train, task.y_train, task.x_test, alpha=cfg.ridge_alpha)
    train_pred = _fit_ridge(task.x_train, task.y_train, task.x_train, alpha=cfg.ridge_alpha)
    residual = task.y_train - train_pred
    latent_dim = max(1, min(int(cfg.latent_dim), residual.shape[1], residual.shape[0]))
    _u, _s, vt = np.linalg.svd(residual, full_matrices=False)
    basis = vt[:latent_dim].T
    train_latent = residual @ basis
    latent_model = _fit_ridge(task.x_train, train_latent, task.x_test, alpha=cfg.ridge_alpha)
    low_rank_residual = latent_model @ basis.T
    adapter = _heldout_subject_adapter(task.y_train, train_pred, task.y_test.shape, scale=cfg.subject_adapter_scale)
    return ridge + low_rank_residual + adapter


def _heldout_subject_adapter(y_train: np.ndarray, train_pred: np.ndarray, target_shape: tuple[int, ...], *, scale: float) -> np.ndarray:
    offset = np.mean(y_train - train_pred, axis=0, keepdims=True) * float(scale)
    return np.broadcast_to(offset, target_shape).astype(np.float64).copy()


def _augment(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return {"mse": float(np.mean(error**2)), "mae": float(np.mean(np.abs(error)))}


def _write_baseline_csv(path: Path, result: dict[str, Any]) -> Path:
    order = list(result["model_order"])
    metrics = result["metrics_by_model"]
    rank_by_model = {row["model_id"]: row["rank"] for row in result["baseline_ranking"]}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model_id", "mse", "mae", "rank", "role"))
        writer.writeheader()
        for model_id in order:
            row = metrics[model_id]
            writer.writerow(
                {
                    "model_id": model_id,
                    "mse": row["mse"],
                    "mae": row["mae"],
                    "rank": rank_by_model[model_id],
                    "role": "candidate" if model_id == "researchdock_observation_operator" else "baseline",
                }
            )
    return path


def _format_observation_report(result: dict[str, Any]) -> str:
    missing_audit = result.get("missing_modality_audit", {})
    split_audit = _observation_split_audit(result)
    lines = [
        "# ResearchDock RD-2 Observation-Model Report",
        "",
        "## Baselines First",
        "",
        "The mean and ridge baselines are evaluated before the ResearchDock observation operator.",
        "",
        "| model | role | mse | mae |",
        "| --- | --- | ---: | ---: |",
    ]
    for model_id in result["model_order"]:
        metrics = result["metrics_by_model"][model_id]
        role = "candidate" if model_id == "researchdock_observation_operator" else "baseline"
        lines.append(f"| {model_id} | {role} | {float(metrics['mse']):.6g} | {float(metrics['mae']):.6g} |")
    split = result.get("split", {})
    if split:
        train_subjects = list(split.get("train_subjects", ()))
        test_subjects = list(split.get("test_subjects", ()))
        lines.extend(
            [
                "",
                "## Subject-Held-Out Split",
                "",
                "- split_type: subject_held_out",
                f"- train_subjects: {len(train_subjects)}",
                f"- test_subjects: {len(test_subjects)}",
                f"- subject_overlap: {bool(split.get('subject_overlap'))}",
            ]
        )
        lines.extend(
            [
                "",
                "## Split Audit",
                "",
                f"- leakage_passed: {split_audit['leakage_passed']}",
                f"- failure_reason_count: {len(split_audit['failure_reasons'])}",
            ]
        )
        if split_audit["failure_reasons"]:
            lines.extend(["", "## Split Audit Failures", ""])
            lines.extend(f"- {reason}" for reason in split_audit["failure_reasons"])
    if missing_audit:
        lines.extend(
            [
                "",
                "## Missing-Modality Audit",
                "",
                f"- total_trials: {missing_audit.get('total_trials')}",
                f"- eligible_trials: {missing_audit.get('eligible_trials')}",
                f"- skipped_trials: {missing_audit.get('skipped_trials')}",
                f"- skip_reasons: {', '.join(missing_audit.get('skip_reasons', ())) or 'none'}",
                "",
                "| reason | count |",
                "| --- | ---: |",
            ]
        )
        for reason in ("missing_sensor_packet", "missing_pupil_diameter", "missing_hrv_proxy", "missing_behavior_response"):
            lines.append(f"| {reason} | {int(missing_audit.get(reason, 0))} |")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- Synthetic pretraining scaffold only; not diagnosis, treatment, or clinical decision support.",
            "- A baseline win is reported honestly and does not create a clinical claim.",
            f"- observation_operator_beats_best_baseline: {result['observation_operator_beats_best_baseline']}",
        ]
    )
    return "\n".join(lines) + "\n"
