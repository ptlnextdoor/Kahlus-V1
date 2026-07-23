"""Shared residual-forecastability evaluation primitives for horizon-shaped gates.

Used by interoception_scout and autonomic_rfs. State-shaped evaluation stays in
passive_pci.evaluate_pci_fixture (different payload shape); leaf helpers here are
shared across both families.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

from neurotwin.forecastability.m1 import (
    _best_baseline,
    _crossfit_proba,
    _crossfit_residual_proba,
    _fit_predict,
    _fit_residual_offset_predict,
    _logistic_factory,
    _moving_average_proba,
    _probe_accuracy,
    _rfs_bits,
    _rfs_payload,
    _within_patient_roll,
)
from neurotwin.forecastability.m4 import _cluster_permutation_rfs


def horizon_payload(
    *,
    y: np.ndarray,
    nuisance_b: np.ndarray,
    z: np.ndarray,
    subject: np.ndarray,
    seed: int,
    horizon: int,
    bootstrap_mode: str,
) -> dict[str, Any]:
    ma = _moving_average_proba(y, subject)
    persistence = np.clip(_within_patient_roll(y.astype(np.float64), subject, shift=1), 1e-5, 1.0 - 1e-5)
    base_rate = np.full(len(y), float(np.mean(y)), dtype=np.float64)
    b_only = _crossfit_proba(nuisance_b, y, subject, _logistic_factory, seed=seed)
    residual = _crossfit_residual_proba(nuisance_b, z, y, subject, seed=seed)
    shuffled = _crossfit_residual_proba(nuisance_b, z, y, subject, seed=seed + 1, control="shuffle")
    shifted = _crossfit_residual_proba(nuisance_b, z, y, subject, seed=seed + 2, control="time_shift")
    surrogate = _crossfit_residual_proba(
        nuisance_b,
        circular_shift_within_subject(z, subject, seed=seed + 33),
        y,
        subject,
        seed=seed + 3,
    )
    best_name, best_baseline = _best_baseline(
        y,
        {
            "moving_average": ma,
            "persistence": persistence,
            "base_rate": base_rate,
            "nuisance_only": b_only,
        },
    )
    residual_payload = _rfs_payload(y, best_baseline, residual, subject, seed=seed + 4, bootstrap_mode=bootstrap_mode)
    perm = _cluster_permutation_rfs(y, best_baseline, residual, subject, seed=seed + 6)
    residual_payload["rfs_p_value_two_sided"] = float(perm.get("p_value", 1.0))
    probe = _probe_accuracy(z, subject)
    return {
        "horizon_epochs": horizon,
        "positive_events": int(np.sum(y)),
        "best_baseline": best_name,
        "residual_model": residual_payload,
        "nested_cv": nested_cv_delta_bits(y, nuisance_b, z, subject),
        "cluster_permutation": perm,
        "controls": {
            "label_shuffle": _rfs_payload(y, best_baseline, shuffled, subject, seed=seed + 8, bootstrap_mode="smoke"),
            "time_shift": _rfs_payload(y, best_baseline, shifted, subject, seed=seed + 9, bootstrap_mode="smoke"),
            "circular_shift_surrogate": _rfs_payload(
                y, best_baseline, surrogate, subject, seed=seed + 10, bootstrap_mode="smoke"
            ),
        },
        "subject_probe": probe,
        "nuisance_probe_failures": probe_failures(probe),
    }


def circular_shift_within_subject(z: np.ndarray, subject: np.ndarray, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = z.copy()
    for group in np.unique(subject):
        idx = np.flatnonzero(subject == group)
        if len(idx) <= 1:
            continue
        shift = int(rng.integers(1, len(idx)))
        out[idx] = np.roll(z[idx], shift, axis=0)
    return out


def nested_cv_delta_bits(
    y: np.ndarray,
    b: np.ndarray,
    z: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    n_splits = min(4, len(set(groups.tolist())))
    if n_splits < 2:
        return {"delta_rfs_bits": 0.0, "delta_rfs_bits_std": 0.0, "n_folds": 0}
    folds = GroupKFold(n_splits=n_splits)
    deltas = []
    for train_idx, test_idx in folds.split(b, y, groups):
        b_train, b_test = b[train_idx], b[test_idx]
        z_train, z_test = z[train_idx], z[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        q0_train = np.clip(_fit_predict(_logistic_factory(), b_train, y_train, b_train), 1e-5, 1.0 - 1e-5)
        q0_test = np.clip(_fit_predict(_logistic_factory(), b_train, y_train, b_test), 1e-5, 1.0 - 1e-5)
        bz_pred = np.clip(
            _fit_residual_offset_predict(z_train, y_train, q0_train, z_test, q0_test),
            1e-5,
            1.0 - 1e-5,
        )
        deltas.append(_rfs_bits(y_test, q0_test, bz_pred))
    return {
        "delta_rfs_bits": float(np.mean(deltas)),
        "delta_rfs_bits_std": float(np.std(deltas)),
        "n_folds": len(deltas),
    }


def probe_failures(probe: dict[str, float]) -> list[str]:
    if probe["accuracy"] > probe["chance"] + 0.2:
        return ["subject_probe_above_chance"]
    return []


def horizon_synthetic_passes(known: dict[str, Any], null: dict[str, Any]) -> bool:
    def _known_ok(payload: dict[str, Any]) -> bool:
        rows = payload["horizons"]
        if not rows:
            return False
        row = rows[0]
        full = row["residual_model"]
        shuffled = row["controls"]["label_shuffle"]
        shifted = row["controls"]["time_shift"]
        return bool(
            payload["positive_events"] >= 40
            and full["rfs_bits"] > 0.02
            and shuffled["rfs_bits"] < full["rfs_bits"] * 0.4
            and shifted["rfs_bits"] < full["rfs_bits"] * 0.4
            and not row["nuisance_probe_failures"]
        )

    def _null_ok(payload: dict[str, Any]) -> bool:
        rows = payload["horizons"]
        if not rows:
            return False
        full = rows[0]["residual_model"]
        return bool(
            payload["positive_events"] >= 20
            and abs(full["rfs_bits"]) < 0.03
            and full["rfs_ci_high"] <= 0.05
        )

    return _known_ok(known) and _null_ok(null)


def horizon_real_passes(
    payload: dict[str, Any] | None,
    *,
    min_subjects: int = 0,
    min_positive_events: int = 0,
) -> bool:
    if payload is None:
        return False
    if payload.get("n_subjects", 0) < min_subjects:
        return False
    rows = payload.get("horizons", [])
    if not rows:
        return False
    primary = rows[0]
    if primary.get("positive_events", 0) < min_positive_events:
        return False
    full = primary["residual_model"]
    controls = primary["controls"]
    return bool(
        full["rfs_bits"] > 0.02
        and full.get("rfs_ci_low", -1.0) > 0.0
        and all(ctrl["rfs_bits"] < full["rfs_bits"] * 0.4 for ctrl in controls.values())
        and not primary["nuisance_probe_failures"]
    )


def write_json(path: Path, payload: Any) -> None:
    def _default(obj: Any) -> Any:
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        raise TypeError

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_default) + "\n",
        encoding="utf-8",
    )
