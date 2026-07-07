from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from neurotwin.researchdock.schemas import ResearchDockSession, ResearchDockTrial

PROFILE_VECTOR_KEYS: tuple[str, ...] = (
    "reward_response_delta",
    "reaction_time_change",
    "effort_persistence_score",
    "pupil_response_amplitude",
    "pupil_recovery_slope",
    "ppg_hrv_proxy_mean",
    "task_accuracy_mean",
    "task_accuracy_std",
)


def compute_researchdock_metrics(sessions: Sequence[ResearchDockSession]) -> list[dict[str, Any]]:
    return [_session_metrics(session) for session in sessions]


def response_profile_vector(metric_row: dict[str, Any]) -> np.ndarray:
    return np.asarray([_finite_float(metric_row.get(key), default=0.0) for key in PROFILE_VECTOR_KEYS], dtype=np.float64)


def audit_response_profile_readiness(
    metric_rows: Sequence[dict[str, Any]],
    *,
    min_sessions: int = 20,
) -> dict[str, Any]:
    """Audit whether response-profile vectors are ready for future clustering.

    This is a readiness audit only. It intentionally does not assign clusters,
    fit latent classes, or make clinical claims.
    """

    rows = tuple(metric_rows)
    vectors = np.asarray([response_profile_vector(row) for row in rows], dtype=np.float64)
    finite_vectors = bool(vectors.size > 0 and np.isfinite(vectors).all())
    quality_counts: dict[str, int] = {}
    for row in rows:
        for flag in row.get("quality_flags", ()):
            quality_counts[str(flag)] = quality_counts.get(str(flag), 0) + 1

    failures: list[str] = []
    if len(rows) < int(min_sessions):
        failures.append("insufficient_sessions_for_future_clustering")
    if not finite_vectors:
        failures.append("nonfinite_profile_vectors")
    if quality_counts.get("missing_pupil", 0):
        failures.append("missing_pupil_profiles_present")

    profile_labels = sorted({str(row.get("profile", "unknown")) for row in rows})
    return {
        "readiness_scope": "future_response_profile_clustering_readiness",
        "clustering_performed": False,
        "ready_for_future_clustering": not failures,
        "n_metric_rows": len(rows),
        "minimum_sessions_required": int(min_sessions),
        "profile_vector_keys": list(PROFILE_VECTOR_KEYS),
        "finite_profile_vectors": finite_vectors,
        "candidate_profile_labels_observed": profile_labels,
        "quality_flag_counts": dict(sorted(quality_counts.items())),
        "failure_reasons": failures,
        "claim_boundary": "readiness_audit_only_no_clustering_no_clinical_claims",
    }


def _session_metrics(session: ResearchDockSession) -> dict[str, Any]:
    trials = tuple(session.trials)
    reward_trials = _trials_by_reward(trials, "reward")
    neutral_trials = _trials_by_reward(trials, "neutral")
    pupil_values = _values(trials, "pupil_diameter")
    reward_pupil = _values(reward_trials, "pupil_diameter")
    neutral_pupil = _values(neutral_trials, "pupil_diameter")
    reward_rt = _values(reward_trials, "reaction_time_ms")
    neutral_rt = _values(neutral_trials, "reaction_time_ms")
    accuracy = _values(trials, "accuracy")
    hrv = _values(trials, "hrv_proxy")
    effort = _values(trials, "effort_level")

    quality_flags = set(session.quality_flags)
    for trial in trials:
        quality_flags.update(trial.quality_flags)
        if trial.sensor_packet is not None:
            quality_flags.update(trial.sensor_packet.quality_flags)
    if pupil_values.size == 0:
        quality_flags.add("missing_pupil")

    return {
        "participant_id_hash": session.participant_id_hash,
        "session_id": session.session_id,
        "profile": str(session.metadata.get("profile", "unknown")),
        "n_trials": len(trials),
        "reward_response_delta": _mean(reward_pupil) - _mean(neutral_pupil),
        "reaction_time_change": _mean(reward_rt) - _mean(neutral_rt),
        "effort_persistence_score": _effort_persistence(effort, accuracy),
        "pupil_response_amplitude": _amplitude(pupil_values),
        "pupil_recovery_slope": _recovery_slope(trials),
        "ppg_hrv_proxy_mean": _mean(hrv),
        "ppg_hrv_proxy_std": _std(hrv),
        "task_accuracy_mean": _mean(accuracy),
        "task_accuracy_std": _std(accuracy),
        "pupil_sample_count": int(pupil_values.size),
        "quality_flags": sorted(quality_flags),
    }


def _trials_by_reward(trials: Sequence[ResearchDockTrial], reward_condition: str) -> tuple[ResearchDockTrial, ...]:
    return tuple(trial for trial in trials if trial.reward_condition == reward_condition)


def _values(trials: Sequence[ResearchDockTrial], field_name: str) -> np.ndarray:
    out: list[float] = []
    for trial in trials:
        value = getattr(trial, field_name, None)
        if value is None and trial.sensor_packet is not None:
            value = getattr(trial.sensor_packet, field_name, None)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            out.append(numeric)
    return np.asarray(out, dtype=np.float64)


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _std(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else 0.0


def _amplitude(values: np.ndarray) -> float:
    return float(np.max(values) - np.min(values)) if values.size else 0.0


def _finite_float(value: object, *, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if np.isfinite(numeric) else default


def _effort_persistence(effort: np.ndarray, accuracy: np.ndarray) -> float:
    if effort.size == 0:
        return _mean(accuracy)
    if accuracy.size == effort.size:
        return float(np.mean((0.5 + effort) * accuracy))
    return float(np.mean(effort))


def _recovery_slope(trials: Sequence[ResearchDockTrial]) -> float:
    rest = [trial for trial in trials if trial.task_name == "recovery_rest"]
    if len(rest) < 2:
        return 0.0
    x = np.asarray([trial.timestamp for trial in rest], dtype=np.float64)
    y = _values(rest, "pupil_diameter")
    if y.size != x.size or y.size < 2:
        return 0.0
    x = x - x[0]
    denom = float(np.dot(x, x))
    if denom == 0.0:
        return 0.0
    return float(np.dot(x, y - y[0]) / denom)
