"""Interoception / peripheral-signal RFS scout on public Sleep-EDF smoke.

Public cousin of Coleman nested delta-CVR2: does peripheral Z add residual
forecastability for sleep-state transitions beyond EEG nuisance B, held-out subject?
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neurotwin.forecastability._rfs_eval import (
    horizon_payload,
    horizon_real_passes,
    horizon_synthetic_passes,
    write_json,
)
from neurotwin.forecastability.m1 import _sigmoid, handcrafted_eeg_features
from neurotwin.forecastability.m2 import (
    _local_sleep_edf_pairs,
    _read_edf_signals,
    _read_sleep_edf_hypnogram,
    _sleep_edf_record_metadata,
    _stage_per_record,
)

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

    synthetic_ok = horizon_synthetic_passes(synthetic_known, synthetic_null)
    if real_status == "evaluated":
        gate_passed = synthetic_ok and horizon_real_passes(real_payload)
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
    write_json(out / "scout_report.json", gate)
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
        horizon_payload(
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
