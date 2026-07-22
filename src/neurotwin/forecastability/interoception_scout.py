"""Interoception / peripheral-signal RFS scout on public Sleep-EDF smoke.

Public cousin of Coleman nested delta-CVR2: does peripheral Z add residual
forecastability for sleep-state transitions beyond EEG nuisance B, held-out subject?
"""
from __future__ import annotations

from dataclasses import dataclass
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
    _sigmoid,
    _within_patient_roll,
    handcrafted_eeg_features,
)
from neurotwin.forecastability.m2 import (
    _local_sleep_edf_pairs,
    _read_edf_signals,
    _read_sleep_edf_hypnogram,
    _sleep_edf_record_metadata,
    _stage_per_record,
)
from neurotwin.forecastability.m4 import _cluster_permutation_rfs

INTEROCEPTION_SCOUT_SCHEMA = "kahlus.forecastability.interoception_scout.v1"
DEFAULT_HORIZONS = (1, 2, 4)
EEG_LABELS = ("EEG Fpz-Cz", "EEG Pz-Oz")
PERIPHERAL_LABELS = ("EOG horizontal", "EMG submental", "Resp oro-nasal")
CLAIM_SCOPE = (
    "peripheral_autonomic_residual_forecastability_sleep_state_transitions_"
    "public_sleep_edf_smoke_not_gastric_egg_not_coleman_data_scout_grade"
)


@dataclass(frozen=True)
class InteroceptionFixture:
    eeg_windows: np.ndarray
    peripheral_windows: np.ndarray
    y_by_horizon: dict[int, np.ndarray]
    subject: np.ndarray
    nuisance: np.ndarray
    base_rate: np.ndarray


def run_interoception_rfs_gate(
    out_dir: str | Path,
    *,
    seed: int = 0,
    sleep_edf_root: str | Path | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    bootstrap_mode: str = "smoke",
    max_pairs: int = 8,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    known = make_interoception_fixture(seed=seed, residual_signal=True, horizons=horizons)
    null = make_interoception_fixture(seed=seed + 100, residual_signal=False, horizons=horizons)
    synthetic_known = _evaluate_fixture(known, seed=seed, horizons=horizons, bootstrap_mode=bootstrap_mode)
    synthetic_null = _evaluate_fixture(null, seed=seed + 100, horizons=horizons, bootstrap_mode=bootstrap_mode)

    real_status = "skipped"
    real_payload: dict[str, Any] | None = None
    real_failures: list[str] = []
    root = Path(sleep_edf_root) if sleep_edf_root is not None else None
    if root is None:
        real_failures.append("sleep_edf_root_missing")
    elif not root.is_dir():
        real_status = "missing"
        real_failures.append("sleep_edf_root_missing")
    else:
        try:
            fixture = load_interoception_sleep_edf_fixture(root, max_pairs=max_pairs, horizons=horizons)
            real_payload = _evaluate_fixture(
                fixture, seed=seed + 200, horizons=horizons, bootstrap_mode=bootstrap_mode
            )
            real_status = "evaluated"
        except Exception as exc:  # noqa: BLE001 - status-typed evidence failure
            real_status = "failed"
            real_failures.append(f"sleep_edf_load_failed:{exc}")

    synthetic_ok = _synthetic_passes(synthetic_known, synthetic_null)
    if real_status == "evaluated":
        gate_passed = synthetic_ok and _real_passes(real_payload)
        stop_reason = (
            "Interoception scout gate passed on synthetic known/null and Sleep-EDF smoke"
            if gate_passed
            else "Interoception scout gate failed on real Sleep-EDF smoke; do not claim peripheral residual forecastability."
        )
    elif real_status in {"failed", "missing"}:
        gate_passed = False
        stop_reason = (
            "Synthetic fixture validated; Sleep-EDF real smoke failed to load — "
            "do not claim peripheral residual forecastability on real data."
        )
    else:
        gate_passed = synthetic_ok
        stop_reason = (
            "Interoception scout gate passed on synthetic known/null (no real Sleep-EDF root)"
            if gate_passed
            else "Interoception scout gate failed; do not claim peripheral residual forecastability."
        )

    gate = {
        "schema": INTEROCEPTION_SCOUT_SCHEMA,
        "milestone": "interoception_rfs_scout",
        "claim_scope": CLAIM_SCOPE,
        "stop_reason": stop_reason,
        "gate_passed": gate_passed,
        "horizons_epochs": list(horizons),
        "synthetic_known": synthetic_known,
        "synthetic_null": synthetic_null,
        "sleep_edf_status": real_status,
        "sleep_edf_real": real_payload,
        "sleep_edf_failures": real_failures,
    }
    _write_json(out / "scout_report.json", gate)
    _write_report(out / "SCOUT_EVIDENCE_REPORT.md", gate)
    return gate


def make_interoception_fixture(
    *,
    seed: int,
    residual_signal: bool,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    n_subjects: int = 12,
    epochs_per_subject: int = 120,
) -> InteroceptionFixture:
    rng = np.random.default_rng(seed)
    eeg_windows: list[np.ndarray] = []
    peripheral_windows: list[np.ndarray] = []
    y_maps: dict[int, list[int]] = {h: [] for h in horizons}
    subjects: list[int] = []
    nuisance: list[list[float]] = []
    base_rates: list[float] = []
    t_axis = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    max_h = max(horizons)
    for subject in range(n_subjects):
        stage = 0
        precursor = 0.0
        raw_eeg: list[np.ndarray] = []
        raw_peripheral: list[np.ndarray] = []
        events = [0]
        for epoch in range(epochs_per_subject + max_h):
            cycle = np.sin(2.0 * np.pi * epoch / 24.0)
            recent = float(np.mean(events[-8:]))
            precursor = 0.8 * precursor + rng.normal(0.0, 0.5)
            visible = precursor if residual_signal else rng.normal(0.0, 1.0)
            eeg = rng.normal(0.0, 0.2, size=(64, 2)).astype(np.float32)
            eeg[:, 0] += 0.5 * cycle * np.sin(2.0 * np.pi * 3.0 * t_axis)
            eeg[:, 1] += 0.3 * np.sin(2.0 * np.pi * 8.0 * t_axis + float(subject))
            peripheral = rng.normal(0.0, 0.15, size=(64, 3)).astype(np.float32)
            peripheral[:, 0] += 1.1 * visible * np.sin(2.0 * np.pi * 2.0 * t_axis)
            peripheral[:, 1] += 0.95 * visible * np.linspace(-1.0, 1.0, 64)
            peripheral[:, 2] += 0.85 * visible * np.cos(2.0 * np.pi * 1.5 * t_axis)
            logit = -2.0 + 0.7 * cycle + 1.1 * recent + (2.5 * precursor if residual_signal else 0.0)
            event = int(rng.random() < _sigmoid(logit))
            raw_eeg.append(eeg)
            raw_peripheral.append(peripheral)
            events.append(event)
            if event:
                stage = int(np.clip(stage + rng.choice([-1, 0, 1]), 0, 4))
        event_arr = np.asarray(events[1:], dtype=np.int64)
        recent_changes = [0]
        for epoch in range(epochs_per_subject):
            for h in horizons:
                y_maps[h].append(int(np.any(event_arr[epoch : epoch + h])))
            eeg_windows.append(raw_eeg[epoch])
            peripheral_windows.append(raw_peripheral[epoch])
            subjects.append(subject)
            recent = float(np.mean(recent_changes[-10:]))
            cycle = np.sin(2.0 * np.pi * epoch / 24.0)
            nuisance.append([1.0, cycle, np.cos(2.0 * np.pi * epoch / 24.0), recent, stage / 4.0])
            base_rates.append(float(np.mean(events[max(0, epoch - 20) : epoch + 1])))
            recent_changes.append(int(events[epoch + 1] != events[epoch]))
    return InteroceptionFixture(
        eeg_windows=np.asarray(eeg_windows, dtype=np.float32),
        peripheral_windows=np.asarray(peripheral_windows, dtype=np.float32),
        y_by_horizon={h: np.asarray(y_maps[h], dtype=np.int64) for h in horizons},
        subject=np.asarray(subjects, dtype=np.int64),
        nuisance=np.asarray(nuisance, dtype=np.float32),
        base_rate=np.asarray(base_rates, dtype=np.float32),
    )


def load_interoception_sleep_edf_fixture(
    root: str | Path,
    *,
    max_pairs: int = 8,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> InteroceptionFixture:
    pairs = _local_sleep_edf_pairs(Path(root))[:max_pairs]
    if len(pairs) < 3:
        raise ValueError("need at least 3 Sleep-EDF pairs")
    metadata = [_sleep_edf_record_metadata(psg) for psg, _hyp in pairs]
    subject_codes = {
        row["subject_id"]: idx for idx, row in enumerate({r["subject_id"]: r for r in metadata}.values())
    }
    eeg_windows: list[np.ndarray] = []
    peripheral_windows: list[np.ndarray] = []
    y_maps: dict[int, list[int]] = {h: [] for h in horizons}
    subjects: list[int] = []
    nuisance: list[list[float]] = []
    base_rates: list[float] = []
    max_h = max(horizons)
    for meta, (psg_path, hyp_path) in zip(metadata, pairs, strict=True):
        eeg_psg = _read_edf_signals(psg_path, preferred_labels=EEG_LABELS, min_channels=2)
        peripheral_psg = _read_edf_signals(psg_path, preferred_labels=PERIPHERAL_LABELS, min_channels=2)
        hyp = _read_sleep_edf_hypnogram(hyp_path)
        labels = _stage_per_record(
            hyp,
            n_records=eeg_psg["signals"].shape[0],
            record_seconds=float(eeg_psg["record_duration"]),
        )
        n = min(eeg_psg["signals"].shape[0], peripheral_psg["signals"].shape[0], len(labels))
        if n <= max_h + 4:
            continue
        eeg = eeg_psg["signals"][:n]
        peripheral = peripheral_psg["signals"][:n]
        labels = labels[:n]
        transitions = (np.diff(labels, prepend=labels[0]) != 0).astype(np.int64)
        subj = subject_codes[meta["subject_id"]]
        recent_changes = [0]
        for epoch in range(n - max_h):
            for h in horizons:
                y_maps[h].append(int(np.any(transitions[epoch + 1 : epoch + h + 1])))
            eeg_windows.append(eeg[epoch])
            peripheral_windows.append(peripheral[epoch])
            subjects.append(subj)
            recent = float(np.mean(recent_changes[-10:]))
            nuisance.append(
                [
                    1.0,
                    np.sin(2.0 * np.pi * epoch / max(1, n)),
                    np.cos(2.0 * np.pi * epoch / max(1, n)),
                    recent,
                    float(labels[epoch]) / 5.0,
                ]
            )
            base_rates.append(float(np.mean(transitions[max(0, epoch - 10) : epoch + 1])))
            recent_changes.append(int(transitions[epoch]))
    if not eeg_windows:
        raise ValueError("no usable Sleep-EDF interoception windows parsed")
    return InteroceptionFixture(
        eeg_windows=np.asarray(eeg_windows, dtype=np.float32),
        peripheral_windows=np.asarray(peripheral_windows, dtype=np.float32),
        y_by_horizon={h: np.asarray(y_maps[h], dtype=np.int64) for h in horizons},
        subject=np.asarray(subjects, dtype=np.int64),
        nuisance=np.asarray(nuisance, dtype=np.float32),
        base_rate=np.asarray(base_rates, dtype=np.float32),
    )


def _evaluate_fixture(
    fixture: InteroceptionFixture,
    *,
    seed: int,
    horizons: tuple[int, ...],
    bootstrap_mode: str,
) -> dict[str, Any]:
    b_matrix = handcrafted_eeg_features(fixture.eeg_windows)
    z_matrix = handcrafted_eeg_features(fixture.peripheral_windows)
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
        "horizons": rows,
        "nested_cv_delta_bits": [row["nested_cv"]["delta_rfs_bits"] for row in rows],
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


def _real_passes(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    rows = payload.get("horizons", [])
    if not rows:
        return False
    primary = rows[0]["residual_model"]
    controls = rows[0]["controls"]
    return bool(
        primary["rfs_bits"] > 0.02
        and primary.get("rfs_ci_low", -1.0) > 0.0
        and all(ctrl["rfs_bits"] < primary["rfs_bits"] * 0.4 for ctrl in controls.values())
        and not rows[0]["nuisance_probe_failures"]
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
        "# Interoception RFS scout",
        "",
        f"- claim_scope: `{gate['claim_scope']}`",
        f"- stop_reason: {gate['stop_reason']}",
        f"- gate_passed: **{gate['gate_passed']}**",
        f"- sleep_edf_status: `{gate['sleep_edf_status']}`",
        "",
        "## Synthetic known / null",
        "",
        f"- known primary RFS bits: {gate['synthetic_known']['horizons'][0]['residual_model']['rfs_bits']:.4f}",
        f"- null primary RFS bits: {gate['synthetic_null']['horizons'][0]['residual_model']['rfs_bits']:.4f}",
        "",
        "## Sleep-EDF real smoke",
        "",
    ]
    if gate["sleep_edf_real"] is None:
        lines.append(f"- not run: {gate['sleep_edf_failures']}")
    else:
        real = gate["sleep_edf_real"]
        lines.append(f"- windows: {real['n_windows']} subjects: {real['n_subjects']}")
        lines.append(
            f"- primary residual RFS bits: {real['horizons'][0]['residual_model']['rfs_bits']:.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
