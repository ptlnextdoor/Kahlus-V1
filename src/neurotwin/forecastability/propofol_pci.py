"""Propofol Passive PCI: awake vs sedated state discrimination on OpenNeuro ds005620."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from neurotwin.forecastability.passive_pci import (
    PassivePciFixture,
    _real_passes,
    _synthetic_passes,
    _write_json,
    evaluate_pci_fixture,
    make_passive_pci_fixture,
)

PROPOFOL_PCI_SCHEMA = "kahlus.forecastability.propofol_pci.v1"
MACROSTATES = ("awake", "sedated")
CLAIM_SCOPE = (
    "propofol_sedation_state_discrimination_complexity_beyond_spectral_baseline_"
    "subject_held_out_openneuro_ds005620_not_tms_pci_not_clinical"
)
EPOCH_SECONDS = 10.0
TARGET_SFREQ = 100.0
PREFERRED_CHANNELS = ("Fz", "Cz", "Pz", "Oz", "C3", "C4", "F3", "F4")
_TASK_RE = re.compile(r"task-([A-Za-z0-9]+)")


def run_propofol_pci_gate(
    out_dir: str | Path,
    *,
    seed: int = 0,
    ds_root: str | Path | None = None,
    bootstrap_mode: str = "smoke",
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    known = make_propofol_pci_fixture(seed=seed, residual_complexity=True)
    null = make_propofol_pci_fixture(seed=seed + 100, residual_complexity=False)
    synthetic_known = evaluate_pci_fixture(known, state_names=MACROSTATES, seed=seed, bootstrap_mode=bootstrap_mode)
    synthetic_null = evaluate_pci_fixture(null, state_names=MACROSTATES, seed=seed + 100, bootstrap_mode=bootstrap_mode)

    real_status = "skipped"
    real_payload: dict[str, Any] | None = None
    real_failures: list[str] = []
    root = Path(ds_root) if ds_root is not None else None
    if root is None:
        real_failures.append("ds005620_root_missing")
    elif not root.is_dir():
        real_status = "missing"
        real_failures.append("ds005620_root_missing")
    else:
        try:
            fixture = load_ds005620_fixture(root)
            real_payload = evaluate_pci_fixture(
                fixture, state_names=MACROSTATES, seed=seed + 200, bootstrap_mode=bootstrap_mode
            )
            real_status = "evaluated"
        except Exception as exc:  # noqa: BLE001 - status-typed evidence failure
            real_status = "failed"
            real_failures.append(f"ds005620_load_failed:{exc}")

    synthetic_ok = _synthetic_passes(synthetic_known, synthetic_null)
    if real_status == "evaluated":
        gate_passed = synthetic_ok and _real_passes(real_payload)
        stop_reason = (
            "Propofol PCI gate passed on synthetic known/null and powered ds005620 cohort"
            if gate_passed
            else "Propofol PCI gate failed on ds005620; do not claim passive complexity beats spectral baseline under propofol sedation."
        )
    elif real_status in {"failed", "missing"}:
        gate_passed = False
        stop_reason = (
            "Synthetic fixture validated; ds005620 real cohort failed to load — "
            "do not claim propofol PCI on real data."
        )
    else:
        gate_passed = synthetic_ok
        stop_reason = (
            "Propofol PCI gate passed on synthetic known/null (no ds005620 root)"
            if gate_passed
            else "Propofol PCI gate failed; do not claim passive complexity beyond spectral baseline."
        )

    gate = {
        "schema": PROPOFOL_PCI_SCHEMA,
        "milestone": "propofol_pci_state",
        "claim_scope": CLAIM_SCOPE,
        "stop_reason": stop_reason,
        "gate_passed": gate_passed,
        "macrostates": list(MACROSTATES),
        "bootstrap_mode": bootstrap_mode,
        "epoch_seconds": EPOCH_SECONDS,
        "target_sfreq": TARGET_SFREQ,
        "synthetic_known": synthetic_known,
        "synthetic_null": synthetic_null,
        "ds005620_status": real_status,
        "ds005620_real": real_payload,
        "ds005620_failures": real_failures,
    }
    _write_json(out / "propofol_pci_report.json", gate)
    _write_propofol_report(out / "PROPOFOL_PCI_EVIDENCE_REPORT.md", gate)
    return gate


def make_propofol_pci_fixture(
    *,
    seed: int,
    residual_complexity: bool,
    n_subjects: int = 12,
    epochs_per_subject: int = 120,
) -> PassivePciFixture:
    rng = np.random.default_rng(seed)
    eeg_windows: list[np.ndarray] = []
    macrostates: list[int] = []
    subjects: list[int] = []
    nuisance: list[list[float]] = []
    t_axis = np.linspace(0.0, 1.0, 100, dtype=np.float32)
    state_cycle = [0, 1, 1, 0, 1]
    for subject in range(n_subjects):
        for epoch in range(epochs_per_subject):
            macro = int(state_cycle[epoch % len(state_cycle)])
            cycle = np.sin(2.0 * np.pi * epoch / 20.0)
            spectral_marker = 0.7 * cycle + 0.15 * float(subject) / max(1, n_subjects)
            complexity_marker = (2.2 * macro if residual_complexity else 0.0) + rng.normal(0.0, 0.12)
            eeg = rng.normal(0.0, 0.1, size=(100, 4)).astype(np.float32)
            eeg[:, 0] += spectral_marker * np.sin(2.0 * np.pi * 4.0 * t_axis)
            eeg[:, 1] += spectral_marker * np.cos(2.0 * np.pi * 6.0 * t_axis)
            if residual_complexity:
                pattern = np.sin(2.0 * np.pi * (3.0 + macro) * t_axis)
                eeg[:, 2] += complexity_marker * pattern
                eeg[:, 3] += complexity_marker * np.roll(pattern, 4)
            eeg_windows.append(eeg)
            macrostates.append(macro)
            subjects.append(subject)
            nuisance.append([1.0, cycle, np.cos(2.0 * np.pi * epoch / 20.0), epoch / max(1, epochs_per_subject)])
    return PassivePciFixture(
        eeg_windows=np.asarray(eeg_windows, dtype=np.float32),
        macrostate=np.asarray(macrostates, dtype=np.int64),
        subject=np.asarray(subjects, dtype=np.int64),
        nuisance=np.asarray(nuisance, dtype=np.float32),
    )


def load_ds005620_fixture(root: str | Path) -> PassivePciFixture:
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("mne is required for ds005620 loading; install neurotwin[moabb]") from exc

    root_path = Path(root)
    participants = root_path / "participants.tsv"
    if not participants.is_file():
        raise FileNotFoundError(f"missing participants.tsv in {root_path}")
    subject_ids = _read_participant_ids(participants)
    subject_codes = {subject_id: idx for idx, subject_id in enumerate(subject_ids)}

    eeg_windows: list[np.ndarray] = []
    macrostates: list[int] = []
    subjects: list[int] = []
    nuisance: list[list[float]] = []

    vhdr_files = sorted(root_path.rglob("*_eeg.vhdr"))
    if len(vhdr_files) < 8:
        raise ValueError("need at least 8 BrainVision EEG files in ds005620")

    for vhdr in vhdr_files:
        subject_id = _subject_from_path(vhdr)
        if subject_id not in subject_codes:
            continue
        task = _task_from_path(vhdr)
        macro = _macrostate_from_task(task)
        if macro is None:
            continue
        raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
        picks = _pick_eeg_channels(raw, max_channels=4)
        raw.pick(picks)
        raw.resample(TARGET_SFREQ, verbose=False)
        n_samples = int(EPOCH_SECONDS * TARGET_SFREQ)
        if raw.n_times < n_samples:
            continue
        data = raw.get_data().T.astype(np.float32)
        n_epochs = data.shape[0] // n_samples
        subj = subject_codes[subject_id]
        for epoch in range(n_epochs):
            window = data[epoch * n_samples : (epoch + 1) * n_samples]
            eeg_windows.append(window)
            macrostates.append(macro)
            subjects.append(subj)
            nuisance.append(
                [
                    1.0,
                    np.sin(2.0 * np.pi * epoch / max(1, n_epochs)),
                    np.cos(2.0 * np.pi * epoch / max(1, n_epochs)),
                    epoch / max(1, n_epochs),
                ]
            )

    if not eeg_windows:
        raise ValueError("no usable ds005620 propofol PCI windows parsed")
    if len(set(subjects)) < 8:
        raise ValueError(f"need at least 8 subjects, found {len(set(subjects))}")
    return PassivePciFixture(
        eeg_windows=np.asarray(eeg_windows, dtype=np.float32),
        macrostate=np.asarray(macrostates, dtype=np.int64),
        subject=np.asarray(subjects, dtype=np.int64),
        nuisance=np.asarray(nuisance, dtype=np.float32),
    )


def _read_participant_ids(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    pid_idx = header.index("participant_id") if "participant_id" in header else 0
    return [line.split("\t")[pid_idx] for line in lines[1:] if line.strip()]


def _subject_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("sub-"):
            return part
    raise ValueError(f"no sub-* in path {path}")


def _task_from_path(path: Path) -> str:
    match = _TASK_RE.search(path.as_posix())
    if match is None:
        raise ValueError(f"no task-* in path {path}")
    return match.group(1)


def _macrostate_from_task(task: str) -> int | None:
    if task == "awake":
        return 0
    if task in {"sed", "sed2"}:
        return 1
    return None


def _pick_eeg_channels(raw: Any, *, max_channels: int) -> list[str]:
    names = raw.ch_names
    selected: list[str] = []
    for wanted in PREFERRED_CHANNELS:
        for name in names:
            if wanted.lower() in name.lower() and name not in selected:
                selected.append(name)
                break
        if len(selected) >= max_channels:
            break
    if len(selected) < 2:
        selected = names[:max_channels]
    return selected[:max_channels]



def _write_propofol_report(path: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# Propofol PCI state-discrimination gate",
        "",
        f"- claim_scope: `{gate['claim_scope']}`",
        f"- stop_reason: {gate['stop_reason']}",
        f"- gate_passed: **{gate['gate_passed']}**",
        f"- ds005620_status: `{gate['ds005620_status']}`",
        f"- bootstrap_mode: `{gate['bootstrap_mode']}`",
        f"- epoch_seconds: {gate['epoch_seconds']}",
        "",
        "## Synthetic known / null",
        "",
        f"- known awake RFS bits: {gate['synthetic_known']['states'][0]['residual_model']['rfs_bits']:.4f}",
        f"- null awake RFS bits: {gate['synthetic_null']['states'][0]['residual_model']['rfs_bits']:.4f}",
        "",
        "## ds005620 real cohort",
        "",
    ]
    if gate["ds005620_real"] is None:
        lines.append(f"- not run: {gate['ds005620_failures']}")
    else:
        real = gate["ds005620_real"]
        lines.append(f"- windows: {real['n_windows']} subjects: {real['n_subjects']}")
        for row in real["states"]:
            lines.append(
                f"- {row['state']}: positive_windows={row['positive_windows']} "
                f"residual_rfs_bits={row['residual_model']['rfs_bits']:.4f} "
                f"ci=[{row['residual_model'].get('rfs_ci_low', float('nan')):.4f}, "
                f"{row['residual_model'].get('rfs_ci_high', float('nan')):.4f}]"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
