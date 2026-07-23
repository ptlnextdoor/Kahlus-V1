"""Autonomic RFS gate — residual peripheral forecastability for arousal beyond cortical spectral.

Flagship NeurIPS defendant: reformulated Y = micro-arousal in horizon h, beyond EEG spectral B.
Supports NSRR MESA (train) and SHHS (dataset-held-out test) when credentialed data is present.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

from neurotwin.adapters.nsrr import (
    discover_nsrr_recordings,
    epoch_arousal_mask,
    load_nsrr_epoch_matrix,
    parse_nsrr_arousal_events,
)
from neurotwin.forecastability.complexity_features import spectral_slope_block
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
    _sigmoid,
    _within_patient_roll,
    handcrafted_eeg_features,
)
from neurotwin.forecastability.m4 import _cluster_permutation_rfs

AUTONOMIC_RFS_SCHEMA = "kahlus.forecastability.autonomic_rfs.v1"
DEFAULT_HORIZONS = (1, 2, 4)
DEFAULT_EPOCH_SECONDS = 30.0
CLAIM_SCOPE = (
    "autonomic_residual_forecastability_micro_arousal_beyond_cortical_spectral_"
    "subject_held_out_nsrr_mesa_shhs_not_gastric_egg_not_coleman_data_not_clinical"
)


@dataclass(frozen=True)
class AutonomicRfsFixture:
    eeg_epochs: np.ndarray
    autonomic_epochs: np.ndarray
    y_by_horizon: dict[int, np.ndarray]
    subject: np.ndarray
    dataset: np.ndarray
    nuisance: np.ndarray
    base_rate: np.ndarray


def run_autonomic_rfs_gate(
    out_dir: str | Path,
    *,
    seed: int = 0,
    mesa_root: str | Path | None = None,
    shhs_root: str | Path | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    bootstrap_mode: str = "smoke",
    max_recordings: int | None = None,
    min_subjects: int = 8,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    known = make_autonomic_fixture(seed=seed, residual_signal=True, horizons=horizons)
    null = make_autonomic_fixture(seed=seed + 100, residual_signal=False, horizons=horizons)
    synthetic_known = _evaluate_fixture(known, seed=seed, horizons=horizons, bootstrap_mode=bootstrap_mode)
    synthetic_null = _evaluate_fixture(null, seed=seed + 100, horizons=horizons, bootstrap_mode=bootstrap_mode)

    mesa_status = "skipped"
    mesa_payload: dict[str, Any] | None = None
    mesa_failures: list[str] = []
    shhs_status = "skipped"
    shhs_payload: dict[str, Any] | None = None
    shhs_failures: list[str] = []

    if mesa_root is not None and Path(mesa_root).is_dir():
        try:
            mesa_fixture = load_nsrr_autonomic_fixture(
                mesa_root,
                dataset_code=0,
                dataset_name="mesa",
                horizons=horizons,
                max_recordings=max_recordings,
            )
            mesa_payload = _evaluate_fixture(
                mesa_fixture, seed=seed + 200, horizons=horizons, bootstrap_mode=bootstrap_mode
            )
            mesa_status = "evaluated"
        except Exception as exc:  # noqa: BLE001
            mesa_status = "failed"
            mesa_failures.append(f"mesa_load_failed:{exc}")
    else:
        mesa_failures.append("mesa_root_missing")

    if shhs_root is not None and Path(shhs_root).is_dir() and mesa_payload is not None:
        try:
            shhs_fixture = load_nsrr_autonomic_fixture(
                shhs_root,
                dataset_code=1,
                dataset_name="shhs",
                horizons=horizons,
                max_recordings=max_recordings,
            )
            shhs_payload = _evaluate_fixture(
                shhs_fixture, seed=seed + 300, horizons=horizons, bootstrap_mode=bootstrap_mode
            )
            shhs_status = "evaluated"
        except Exception as exc:  # noqa: BLE001
            shhs_status = "failed"
            shhs_failures.append(f"shhs_load_failed:{exc}")
    elif shhs_root is None or not Path(shhs_root).is_dir():
        shhs_failures.append("shhs_root_missing")

    synthetic_ok = _synthetic_passes(synthetic_known, synthetic_null)
    mesa_ok = _real_passes(mesa_payload, min_subjects=min_subjects)
    if mesa_status == "evaluated":
        if mesa_ok:
            gate_passed = synthetic_ok and mesa_ok
            stop_reason = (
                "Autonomic RFS gate passed on synthetic known/null and powered MESA cohort"
                if gate_passed
                else "Autonomic RFS gate failed powered criteria on MESA; do not claim autonomic residual forecastability."
            )
        else:
            gate_passed = False
            stop_reason = (
                "Autonomic RFS honest negative on MESA under held-out evaluation; "
                "do not claim autonomic residual forecastability beyond cortical spectral baseline."
            )
    elif mesa_status in {"failed", "skipped"}:
        gate_passed = synthetic_ok and mesa_status != "failed"
        stop_reason = (
            "Synthetic fixture validated; MESA real cohort not available — "
            "obtain NSRR credentialed access before claim-grade autonomic RFS."
            if mesa_status == "skipped"
            else "MESA load failed; do not claim autonomic residual forecastability on real data."
        )
    else:
        gate_passed = synthetic_ok
        stop_reason = "Synthetic-only evaluation."

    gate = {
        "schema": AUTONOMIC_RFS_SCHEMA,
        "milestone": "autonomic_rfs_arousal",
        "claim_scope": CLAIM_SCOPE,
        "stop_reason": stop_reason,
        "gate_passed": gate_passed,
        "horizons_epochs": list(horizons),
        "epoch_seconds": DEFAULT_EPOCH_SECONDS,
        "bootstrap_mode": bootstrap_mode,
        "min_subjects": min_subjects,
        "synthetic_known": synthetic_known,
        "synthetic_null": synthetic_null,
        "mesa_status": mesa_status,
        "mesa_real": mesa_payload,
        "mesa_failures": mesa_failures,
        "shhs_status": shhs_status,
        "shhs_dataset_held_out": shhs_payload,
        "shhs_failures": shhs_failures,
        "estimand": {
            "y": "micro_arousal_within_horizon_epochs",
            "b": "cortical_spectral_plus_cycle_history",
            "z": "autonomic_hrv_resp_eog_emg_infraslow",
            "split": "subject_held_out_primary_dataset_held_out_secondary",
        },
    }
    _write_json(out / "autonomic_rfs_report.json", gate)
    _write_report(out / "AUTONOMIC_RFS_EVIDENCE_REPORT.md", gate)
    return gate


def make_autonomic_fixture(
    *,
    seed: int,
    residual_signal: bool,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    n_subjects: int = 16,
    epochs_per_subject: int = 180,
) -> AutonomicRfsFixture:
    rng = np.random.default_rng(seed)
    eeg_epochs: list[np.ndarray] = []
    autonomic_epochs: list[np.ndarray] = []
    y_maps: dict[int, list[int]] = {h: [] for h in horizons}
    subjects: list[int] = []
    datasets: list[int] = []
    nuisance: list[list[float]] = []
    base_rates: list[float] = []
    max_h = max(horizons)
    samples = 128
    t_axis = np.linspace(0.0, 1.0, samples, dtype=np.float32)
    for subject in range(n_subjects):
        dataset_code = 0 if subject % 3 else 1
        precursor = 0.0
        arousal_events: list[int] = [0]
        raw_eeg: list[np.ndarray] = []
        raw_auto: list[np.ndarray] = []
        for epoch in range(epochs_per_subject + max_h):
            cycle = np.sin(2.0 * np.pi * epoch / 30.0)
            precursor = 0.85 * precursor + rng.normal(0.0, 0.4)
            visible = precursor if residual_signal else rng.normal(0.0, 1.0)
            eeg = rng.normal(0.0, 0.15, size=(samples, 1)).astype(np.float32)
            eeg[:, 0] += 0.6 * cycle * np.sin(2.0 * np.pi * 4.0 * t_axis)
            auto = rng.normal(0.0, 0.12, size=(samples, 4)).astype(np.float32)
            auto[:, 0] += 1.2 * visible * np.sin(2.0 * np.pi * 1.5 * t_axis)
            auto[:, 1] += 0.9 * visible * np.cos(2.0 * np.pi * 0.8 * t_axis)
            auto[:, 2] += 0.7 * visible * np.linspace(-1.0, 1.0, samples)
            auto[:, 3] += 0.8 * visible * np.sin(2.0 * np.pi * 2.2 * t_axis)
            logit = -2.2 + 0.8 * cycle + 1.4 * float(np.mean(arousal_events[-6:])) + (2.8 * precursor if residual_signal else 0.0)
            event = int(rng.random() < _sigmoid(logit))
            raw_eeg.append(eeg)
            raw_auto.append(auto)
            arousal_events.append(event)
        event_arr = np.asarray(arousal_events[1:], dtype=np.int64)
        recent_changes = [0]
        for epoch in range(epochs_per_subject):
            for h in horizons:
                y_maps[h].append(int(np.any(event_arr[epoch + 1 : epoch + h + 1])))
            eeg_epochs.append(raw_eeg[epoch])
            autonomic_epochs.append(raw_auto[epoch])
            subjects.append(subject)
            datasets.append(dataset_code)
            recent = float(np.mean(recent_changes[-12:]))
            nuisance.append([1.0, cycle, np.cos(2.0 * np.pi * epoch / 30.0), recent, float(subject) / max(1, n_subjects)])
            base_rates.append(float(np.mean(arousal_events[max(0, epoch - 15) : epoch + 1])))
            recent_changes.append(int(arousal_events[epoch + 1] != arousal_events[epoch]))
    return AutonomicRfsFixture(
        eeg_epochs=np.asarray(eeg_epochs, dtype=np.float32),
        autonomic_epochs=np.asarray(autonomic_epochs, dtype=np.float32),
        y_by_horizon={h: np.asarray(y_maps[h], dtype=np.int64) for h in horizons},
        subject=np.asarray(subjects, dtype=np.int64),
        dataset=np.asarray(datasets, dtype=np.int64),
        nuisance=np.asarray(nuisance, dtype=np.float32),
        base_rate=np.asarray(base_rates, dtype=np.float32),
    )


def load_nsrr_autonomic_fixture(
    root: str | Path,
    *,
    dataset_code: int,
    dataset_name: str,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    max_recordings: int | None = None,
    epoch_seconds: float = DEFAULT_EPOCH_SECONDS,
) -> AutonomicRfsFixture:
    recordings = discover_nsrr_recordings(root, dataset=dataset_name)
    if max_recordings is not None:
        recordings = recordings[:max_recordings]
    if len(recordings) < 3:
        raise ValueError(f"need at least 3 NSRR recordings in {root}, found {len(recordings)}")
    subject_codes = {rec.subject_id: idx for idx, rec in enumerate({r.subject_id: r for r in recordings}.values())}
    eeg_epochs: list[np.ndarray] = []
    autonomic_epochs: list[np.ndarray] = []
    y_maps: dict[int, list[int]] = {h: [] for h in horizons}
    subjects: list[int] = []
    datasets: list[int] = []
    nuisance: list[list[float]] = []
    base_rates: list[float] = []
    max_h = max(horizons)
    for rec in recordings:
        events = parse_nsrr_arousal_events(rec.xml_path)
        matrix = load_nsrr_epoch_matrix(rec.edf_path, epoch_seconds=epoch_seconds)
        n_epochs = matrix["n_epochs"]
        arousal_mask = epoch_arousal_mask(events, n_epochs=n_epochs, epoch_seconds=epoch_seconds)
        eeg_stack = _stack_epoch_channels(matrix["epochs"], ("eeg",))
        auto_stack = _stack_epoch_channels(matrix["epochs"], ("ecg", "eog", "emg", "resp"))
        if eeg_stack.shape[0] < max_h + 4:
            continue
        subj = subject_codes[rec.subject_id]
        recent_changes = [0]
        for epoch in range(n_epochs - max_h):
            for h in horizons:
                y_maps[h].append(int(np.any(arousal_mask[epoch + 1 : epoch + h + 1])))
            eeg_epochs.append(_resample_epoch(eeg_stack[epoch], target_len=128))
            autonomic_epochs.append(_resample_epoch(auto_stack[epoch], target_len=128))
            subjects.append(subj)
            datasets.append(dataset_code)
            recent = float(np.mean(recent_changes[-12:]))
            nuisance.append(
                [
                    1.0,
                    np.sin(2.0 * np.pi * epoch / max(1, n_epochs)),
                    np.cos(2.0 * np.pi * epoch / max(1, n_epochs)),
                    recent,
                    float(arousal_mask[epoch]),
                ]
            )
            base_rates.append(float(np.mean(arousal_mask[max(0, epoch - 10) : epoch + 1])))
            recent_changes.append(int(arousal_mask[epoch] != arousal_mask[max(0, epoch - 1)]))
    if not eeg_epochs:
        raise ValueError(f"no usable NSRR autonomic windows parsed from {root}")
    if len(set(subjects)) < 3:
        raise ValueError("need at least 3 subjects with usable NSRR windows")
    return AutonomicRfsFixture(
        eeg_epochs=np.asarray(eeg_epochs, dtype=np.float32),
        autonomic_epochs=np.asarray(autonomic_epochs, dtype=np.float32),
        y_by_horizon={h: np.asarray(y_maps[h], dtype=np.int64) for h in horizons},
        subject=np.asarray(subjects, dtype=np.int64),
        dataset=np.asarray(datasets, dtype=np.int64),
        nuisance=np.asarray(nuisance, dtype=np.float32),
        base_rate=np.asarray(base_rates, dtype=np.float32),
    )


def _stack_epoch_channels(epochs: dict[str, np.ndarray], keys: tuple[str, ...]) -> np.ndarray:
    parts = [epochs[key] for key in keys if key in epochs]
    if not parts:
        raise ValueError(f"missing epoch channels for keys {keys}")
    stacked = np.concatenate(parts, axis=1)
    return stacked


def _resample_epoch(epoch: np.ndarray, *, target_len: int) -> np.ndarray:
    if epoch.shape[0] == target_len:
        return epoch.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, epoch.shape[0])
    x_new = np.linspace(0.0, 1.0, target_len)
    out = np.zeros((target_len, epoch.shape[1]), dtype=np.float32)
    for ch in range(epoch.shape[1]):
        out[:, ch] = np.interp(x_new, x_old, epoch[:, ch])
    return out


def autonomic_feature_block(epochs: np.ndarray) -> np.ndarray:
    """HRV/resp/EOG/EMG infraslow features per epoch window."""
    n = epochs.shape[0]
    feats: list[list[float]] = []
    for idx in range(n):
        window = epochs[idx]
        row: list[float] = []
        for ch in range(window.shape[1]):
            signal = window[:, ch].astype(np.float64)
            diff = np.diff(signal)
            row.extend(
                [
                    float(np.std(signal)),
                    float(np.mean(np.abs(diff))),
                    float(np.percentile(signal, 90) - np.percentile(signal, 10)),
                    float(np.sqrt(np.mean(diff**2))) if diff.size else 0.0,
                ]
            )
        feats.append(row)
    return np.asarray(feats, dtype=np.float32)


def _evaluate_fixture(
    fixture: AutonomicRfsFixture,
    *,
    seed: int,
    horizons: tuple[int, ...],
    bootstrap_mode: str,
) -> dict[str, Any]:
    eeg_3d = fixture.eeg_epochs[:, :, None] if fixture.eeg_epochs.ndim == 2 else fixture.eeg_epochs
    b_matrix = np.concatenate([handcrafted_eeg_features(eeg_3d), spectral_slope_block(eeg_3d)], axis=1)
    z_matrix = autonomic_feature_block(fixture.autonomic_epochs)
    nuisance_b = np.concatenate([fixture.nuisance, fixture.base_rate[:, None], b_matrix], axis=1)
    rows = [
        _horizon_payload(
            y=fixture.y_by_horizon[h],
            nuisance_b=nuisance_b,
            z=z_matrix,
            subject=fixture.subject,
            seed=seed + h,
            horizon=h,
            bootstrap_mode=bootstrap_mode,
        )
        for h in horizons
    ]
    return {
        "positive_events": int(sum(int(np.sum(fixture.y_by_horizon[h])) for h in horizons)),
        "n_windows": int(len(fixture.subject)),
        "n_subjects": int(len(set(fixture.subject.tolist()))),
        "n_datasets": int(len(set(fixture.dataset.tolist()))),
        "horizons": rows,
    }


def _horizon_payload(
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
        _circular_shift_within_subject(z, subject, seed=seed + 33),
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
        "nested_cv": _nested_cv_delta_bits(y, nuisance_b, z, subject),
        "cluster_permutation": perm,
        "controls": {
            "label_shuffle": _rfs_payload(y, best_baseline, shuffled, subject, seed=seed + 8, bootstrap_mode="smoke"),
            "time_shift": _rfs_payload(y, best_baseline, shifted, subject, seed=seed + 9, bootstrap_mode="smoke"),
            "circular_shift_surrogate": _rfs_payload(
                y, best_baseline, surrogate, subject, seed=seed + 10, bootstrap_mode="smoke"
            ),
        },
        "subject_probe": probe,
        "nuisance_probe_failures": _probe_failures(probe),
    }


def _circular_shift_within_subject(z: np.ndarray, subject: np.ndarray, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = z.copy()
    for group in np.unique(subject):
        idx = np.flatnonzero(subject == group)
        if len(idx) <= 1:
            continue
        shift = int(rng.integers(1, len(idx)))
        out[idx] = np.roll(z[idx], shift, axis=0)
    return out


def _nested_cv_delta_bits(
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
        b_pred = q0_test
        bz_pred = np.clip(
            _fit_residual_offset_predict(z_train, y_train, q0_train, z_test, q0_test),
            1e-5,
            1.0 - 1e-5,
        )
        deltas.append(_rfs_bits(y_test, b_pred, bz_pred))
    return {
        "delta_rfs_bits": float(np.mean(deltas)),
        "delta_rfs_bits_std": float(np.std(deltas)),
        "n_folds": len(deltas),
    }


def _probe_failures(probe: dict[str, float]) -> list[str]:
    if probe["accuracy"] > probe["chance"] + 0.2:
        return ["subject_probe_above_chance"]
    return []


def _synthetic_passes(known: dict[str, Any], null: dict[str, Any]) -> bool:
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


def _real_passes(payload: dict[str, Any] | None, *, min_subjects: int) -> bool:
    if payload is None:
        return False
    if payload.get("n_subjects", 0) < min_subjects:
        return False
    rows = payload.get("horizons", [])
    if not rows:
        return False
    primary = rows[0]
    if primary.get("positive_events", 0) < 100:
        return False
    full = primary["residual_model"]
    controls = primary["controls"]
    return bool(
        full["rfs_bits"] > 0.02
        and full.get("rfs_ci_low", -1.0) > 0.0
        and all(ctrl["rfs_bits"] < full["rfs_bits"] * 0.4 for ctrl in controls.values())
        and not primary["nuisance_probe_failures"]
    )


def _write_json(path: Path, payload: Any) -> None:
    def _default(obj: Any) -> Any:
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        raise TypeError

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_default) + "\n",
        encoding="utf-8",
    )


def _write_report(path: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# Autonomic RFS arousal gate",
        "",
        f"- claim_scope: `{gate['claim_scope']}`",
        f"- stop_reason: {gate['stop_reason']}",
        f"- gate_passed: **{gate['gate_passed']}**",
        f"- mesa_status: `{gate['mesa_status']}`",
        f"- shhs_status: `{gate['shhs_status']}`",
        f"- bootstrap_mode: `{gate['bootstrap_mode']}`",
        f"- epoch_seconds: {gate['epoch_seconds']}",
        "",
        "## Estimand",
        "",
        f"- Y: {gate['estimand']['y']}",
        f"- B: {gate['estimand']['b']}",
        f"- Z: {gate['estimand']['z']}",
        "",
        "## Synthetic known / null",
        "",
        f"- known primary RFS bits: {gate['synthetic_known']['horizons'][0]['residual_model']['rfs_bits']:.4f}",
        f"- null primary RFS bits: {gate['synthetic_null']['horizons'][0]['residual_model']['rfs_bits']:.4f}",
        "",
        "## MESA real cohort",
        "",
    ]
    if gate["mesa_real"] is None:
        lines.append(f"- not run: {gate['mesa_failures']}")
    else:
        real = gate["mesa_real"]
        lines.append(f"- windows: {real['n_windows']} · subjects: {real['n_subjects']}")
        row = real["horizons"][0]
        lines.append(
            f"- h={row['horizon_epochs']}: positive_events={row['positive_events']} "
            f"residual_rfs_bits={row['residual_model']['rfs_bits']:.4f} "
            f"ci=[{row['residual_model'].get('rfs_ci_low', float('nan')):.4f}, "
            f"{row['residual_model'].get('rfs_ci_high', float('nan')):.4f}]"
        )
    lines.extend(["", "## SHHS dataset-held-out", ""])
    if gate["shhs_dataset_held_out"] is None:
        lines.append(f"- not run: {gate['shhs_failures']}")
    else:
        shhs = gate["shhs_dataset_held_out"]
        row = shhs["horizons"][0]
        lines.append(
            f"- windows: {shhs['n_windows']} · subjects: {shhs['n_subjects']} · "
            f"RFS bits: {row['residual_model']['rfs_bits']:.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
