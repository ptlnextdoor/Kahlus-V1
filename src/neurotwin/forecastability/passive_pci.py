"""Passive PCI: subject-held-out sleep-state discrimination via residual complexity RFS."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

from neurotwin.forecastability.complexity_features import complexity_block, spectral_slope_block
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

PASSIVE_PCI_SCHEMA = "kahlus.forecastability.passive_pci.v1"
EEG_LABELS = ("EEG Fpz-Cz", "EEG Pz-Oz")
MACROSTATES = ("wake", "nrem", "rem")
CLAIM_SCOPE = (
    "passive_pci_sleep_state_discrimination_complexity_beyond_spectral_baseline_"
    "subject_held_out_public_sleep_edf_cassette_not_tms_pci_not_clinical"
)


@dataclass(frozen=True)
class PassivePciFixture:
    eeg_windows: np.ndarray
    macrostate: np.ndarray
    subject: np.ndarray
    nuisance: np.ndarray


def run_passive_pci_gate(
    out_dir: str | Path,
    *,
    seed: int = 0,
    sleep_edf_root: str | Path | None = None,
    bootstrap_mode: str = "smoke",
    max_pairs: int | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    known = make_passive_pci_fixture(seed=seed, residual_complexity=True)
    null = make_passive_pci_fixture(seed=seed + 100, residual_complexity=False)
    synthetic_known = _evaluate_fixture(known, seed=seed, bootstrap_mode=bootstrap_mode)
    synthetic_null = _evaluate_fixture(null, seed=seed + 100, bootstrap_mode=bootstrap_mode)

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
            fixture = load_passive_pci_fixture(root, max_pairs=max_pairs)
            real_payload = _evaluate_fixture(fixture, seed=seed + 200, bootstrap_mode=bootstrap_mode)
            real_status = "evaluated"
        except Exception as exc:  # noqa: BLE001 - status-typed evidence failure
            real_status = "failed"
            real_failures.append(f"sleep_edf_load_failed:{exc}")

    synthetic_ok = _synthetic_passes(synthetic_known, synthetic_null)
    if real_status == "evaluated":
        gate_passed = synthetic_ok and _real_passes(real_payload)
        stop_reason = (
            "Passive PCI gate passed on synthetic known/null and powered Sleep-EDF cassette"
            if gate_passed
            else "Passive PCI gate failed on real Sleep-EDF cassette; do not claim passive complexity beats spectral baseline."
        )
    elif real_status in {"failed", "missing"}:
        gate_passed = False
        stop_reason = (
            "Synthetic fixture validated; Sleep-EDF real cohort failed to load — "
            "do not claim passive PCI on real data."
        )
    else:
        gate_passed = synthetic_ok
        stop_reason = (
            "Passive PCI gate passed on synthetic known/null (no real Sleep-EDF root)"
            if gate_passed
            else "Passive PCI gate failed; do not claim passive complexity beyond spectral baseline."
        )

    gate = {
        "schema": PASSIVE_PCI_SCHEMA,
        "milestone": "passive_pci_state",
        "claim_scope": CLAIM_SCOPE,
        "stop_reason": stop_reason,
        "gate_passed": gate_passed,
        "macrostates": list(MACROSTATES),
        "bootstrap_mode": bootstrap_mode,
        "synthetic_known": synthetic_known,
        "synthetic_null": synthetic_null,
        "sleep_edf_status": real_status,
        "sleep_edf_real": real_payload,
        "sleep_edf_failures": real_failures,
    }
    _write_json(out / "passive_pci_report.json", gate)
    _write_report(out / "PASSIVE_PCI_EVIDENCE_REPORT.md", gate)
    return gate


def make_passive_pci_fixture(
    *,
    seed: int,
    residual_complexity: bool,
    n_subjects: int = 12,
    epochs_per_subject: int = 150,
) -> PassivePciFixture:
    rng = np.random.default_rng(seed)
    eeg_windows: list[np.ndarray] = []
    macrostates: list[int] = []
    subjects: list[int] = []
    nuisance: list[list[float]] = []
    t_axis = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    state_cycle = [0, 1, 1, 1, 2, 1, 0]
    for subject in range(n_subjects):
        for epoch in range(epochs_per_subject):
            macro = int(state_cycle[epoch % len(state_cycle)])
            cycle = np.sin(2.0 * np.pi * epoch / 24.0)
            spectral_marker = 0.8 * cycle + 0.2 * float(subject) / max(1, n_subjects)
            complexity_marker = (2.5 * macro if residual_complexity else 0.0) + rng.normal(0.0, 0.15)
            eeg = rng.normal(0.0, 0.1, size=(64, 2)).astype(np.float32)
            eeg[:, 0] += spectral_marker * np.sin(2.0 * np.pi * 3.0 * t_axis)
            eeg[:, 1] += spectral_marker * np.cos(2.0 * np.pi * 5.0 * t_axis)
            if residual_complexity:
                pattern = np.sin(2.0 * np.pi * (4.0 + macro) * t_axis)
                eeg[:, 0] += complexity_marker * pattern
                eeg[:, 1] += complexity_marker * np.roll(pattern, 3)
            else:
                eeg += rng.normal(0.0, 0.12, size=eeg.shape).astype(np.float32)
            eeg_windows.append(eeg)
            macrostates.append(macro)
            subjects.append(subject)
            nuisance.append([1.0, cycle, np.cos(2.0 * np.pi * epoch / 24.0), epoch / max(1, epochs_per_subject)])
    return PassivePciFixture(
        eeg_windows=np.asarray(eeg_windows, dtype=np.float32),
        macrostate=np.asarray(macrostates, dtype=np.int64),
        subject=np.asarray(subjects, dtype=np.int64),
        nuisance=np.asarray(nuisance, dtype=np.float32),
    )


def load_passive_pci_fixture(
    root: str | Path,
    *,
    max_pairs: int | None = None,
) -> PassivePciFixture:
    pairs = _local_sleep_edf_pairs(Path(root))
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    if len(pairs) < 8:
        raise ValueError("need at least 8 Sleep-EDF pairs for powered Passive PCI")
    metadata = [_sleep_edf_record_metadata(psg) for psg, _hyp in pairs]
    subject_codes = {
        row["subject_id"]: idx for idx, row in enumerate({r["subject_id"]: r for r in metadata}.values())
    }
    eeg_windows: list[np.ndarray] = []
    macrostates: list[int] = []
    subjects: list[int] = []
    nuisance: list[list[float]] = []
    for meta, (psg_path, hyp_path) in zip(metadata, pairs, strict=True):
        eeg_psg = _read_edf_signals(psg_path, preferred_labels=EEG_LABELS, min_channels=2)
        hyp = _read_sleep_edf_hypnogram(hyp_path)
        labels = _stage_per_record(
            hyp,
            n_records=eeg_psg["signals"].shape[0],
            record_seconds=float(eeg_psg["record_duration"]),
        )
        n = min(eeg_psg["signals"].shape[0], len(labels))
        if n <= 4:
            continue
        eeg = eeg_psg["signals"][:n]
        labels = labels[:n]
        subj = subject_codes[meta["subject_id"]]
        for epoch in range(n):
            macro = _macrostate_from_stage(int(labels[epoch]))
            if macro is None:
                continue
            eeg_windows.append(eeg[epoch])
            macrostates.append(macro)
            subjects.append(subj)
            nuisance.append(
                [
                    1.0,
                    np.sin(2.0 * np.pi * epoch / max(1, n)),
                    np.cos(2.0 * np.pi * epoch / max(1, n)),
                    epoch / max(1, n),
                ]
            )
    if not eeg_windows:
        raise ValueError("no usable Sleep-EDF Passive PCI windows parsed")
    return PassivePciFixture(
        eeg_windows=np.asarray(eeg_windows, dtype=np.float32),
        macrostate=np.asarray(macrostates, dtype=np.int64),
        subject=np.asarray(subjects, dtype=np.int64),
        nuisance=np.asarray(nuisance, dtype=np.float32),
    )


def _macrostate_from_stage(stage: int) -> int | None:
    if stage == 0:
        return 0
    if stage in {1, 2, 3, 4}:
        return 1
    if stage == 5:
        return 2
    return None


def _evaluate_fixture(
    fixture: PassivePciFixture,
    *,
    seed: int,
    bootstrap_mode: str,
) -> dict[str, Any]:
    spectral = np.concatenate([handcrafted_eeg_features(fixture.eeg_windows), spectral_slope_block(fixture.eeg_windows)], axis=1)
    complexity = complexity_block(fixture.eeg_windows)
    states = []
    for state_idx, state_name in enumerate(MACROSTATES):
        y = (fixture.macrostate == state_idx).astype(np.int64)
        base_rate = np.full(len(y), float(np.mean(y)), dtype=np.float32)
        nuisance_b = np.concatenate([fixture.nuisance, base_rate[:, None], spectral], axis=1)
        states.append(
            _evaluate_state(
                state_name=state_name,
                y=y,
                nuisance_b=nuisance_b,
                z=complexity,
                subject=fixture.subject,
                seed=seed + state_idx * 17,
                bootstrap_mode=bootstrap_mode,
            )
        )
    return {
        "n_windows": int(len(fixture.subject)),
        "n_subjects": int(len(set(fixture.subject.tolist()))),
        "positive_windows_by_state": {
            state_name: int(np.sum(fixture.macrostate == state_idx))
            for state_idx, state_name in enumerate(MACROSTATES)
        },
        "states": states,
    }


def _evaluate_state(
    *,
    state_name: str,
    y: np.ndarray,
    nuisance_b: np.ndarray,
    z: np.ndarray,
    subject: np.ndarray,
    seed: int,
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
    held_out_subjects = _held_out_subject_count(subject)
    return {
        "state": state_name,
        "positive_windows": int(np.sum(y)),
        "n_held_out_subjects": held_out_subjects,
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


def _held_out_subject_count(subject: np.ndarray) -> int:
    unique = np.unique(subject)
    if unique.size <= 1:
        return 0
    folds = GroupKFold(n_splits=min(4, unique.size))
    counts: list[int] = []
    dummy = np.zeros(len(subject), dtype=np.int64)
    for _train, test in folds.split(dummy, dummy, subject):
        counts.append(len(set(subject[test].tolist())))
    return int(max(counts))


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
        rows = payload["states"]
        if not rows:
            return False
        primary = rows[0]["residual_model"]
        shuffled = rows[0]["controls"]["label_shuffle"]
        shifted = rows[0]["controls"]["time_shift"]
        return bool(
            rows[0]["positive_windows"] >= 40
            and primary["rfs_bits"] > 0.02
            and shuffled["rfs_bits"] < primary["rfs_bits"] * 0.4
            and shifted["rfs_bits"] < primary["rfs_bits"] * 0.4
            and not rows[0]["nuisance_probe_failures"]
        )

    def _null_ok(payload: dict[str, Any]) -> bool:
        rows = payload["states"]
        if not rows:
            return False
        primary = rows[0]["residual_model"]
        return bool(
            rows[0]["positive_windows"] >= 20
            and abs(primary["rfs_bits"]) < 0.03
            and primary["rfs_ci_high"] <= 0.05
        )

    return _known_ok(known) and _null_ok(null)


def _real_passes(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    rows = payload.get("states", [])
    if not rows:
        return False
    if payload.get("n_subjects", 0) < 8:
        return False
    powered = [row for row in rows if row["positive_windows"] >= 100]
    if not powered:
        return False
    for row in powered:
        primary = row["residual_model"]
        controls = row["controls"]
        if not (
            primary["rfs_bits"] > 0.02
            and primary.get("rfs_ci_low", -1.0) > 0.0
            and all(ctrl["rfs_bits"] < primary["rfs_bits"] * 0.4 for ctrl in controls.values())
            and not row["nuisance_probe_failures"]
        ):
            return False
    return True


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
        "# Passive PCI state-discrimination gate",
        "",
        f"- claim_scope: `{gate['claim_scope']}`",
        f"- stop_reason: {gate['stop_reason']}",
        f"- gate_passed: **{gate['gate_passed']}**",
        f"- sleep_edf_status: `{gate['sleep_edf_status']}`",
        f"- bootstrap_mode: `{gate['bootstrap_mode']}`",
        "",
        "## Synthetic known / null",
        "",
        f"- known wake RFS bits: {gate['synthetic_known']['states'][0]['residual_model']['rfs_bits']:.4f}",
        f"- null wake RFS bits: {gate['synthetic_null']['states'][0]['residual_model']['rfs_bits']:.4f}",
        "",
        "## Sleep-EDF real cohort",
        "",
    ]
    if gate["sleep_edf_real"] is None:
        lines.append(f"- not run: {gate['sleep_edf_failures']}")
    else:
        real = gate["sleep_edf_real"]
        lines.append(f"- windows: {real['n_windows']} subjects: {real['n_subjects']}")
        for row in real["states"]:
            lines.append(
                f"- {row['state']}: positive_windows={row['positive_windows']} "
                f"residual_rfs_bits={row['residual_model']['rfs_bits']:.4f} "
                f"ci=[{row['residual_model'].get('rfs_ci_low', float('nan')):.4f}, "
                f"{row['residual_model'].get('rfs_ci_high', float('nan')):.4f}]"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
