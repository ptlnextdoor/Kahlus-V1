from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

from neurotwin.forecastability.m1 import (
    TransitionFixture,
    _run_fixture,
)
from neurotwin.models.baselines import NumpyRidgeBaseline
from neurotwin.scoring.metrics import mse, r2_score


SLEEP_EDF_BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/"


@dataclass(frozen=True)
class SleepFixture:
    windows: np.ndarray
    stages: np.ndarray
    transition: np.ndarray
    subject: np.ndarray
    session: np.ndarray
    dataset: np.ndarray
    site: np.ndarray
    nuisance: np.ndarray


def run_m2_gate(out_dir: str | Path, *, seed: int = 0, sleep_edf_root: str | Path | None = None) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fixture = make_synthetic_sleep_fixture(seed=seed)
    synthetic = {
        "source": "synthetic_sleep_fixture",
        "leakage_audit": _sleep_leakage_audit(label_standard="synthetic_stage_labels"),
        "transition_hazard": _run_fixture(_as_transition_fixture(fixture), seed=seed),
        "held_out_channel_reconstruction": _held_out_channel_reconstruction(fixture),
        "cross_dataset_transfer": _cross_dataset_transfer(fixture, seed=seed),
    }
    real_data = sleep_edf_manifest_audit(sleep_edf_root)
    synthetic_passed = _synthetic_sleep_passes(synthetic)
    gate = {
        "milestone": "M2",
        "sleep_edf_source": {
            "name": "Sleep-EDF Expanded",
            "url": SLEEP_EDF_BASE_URL,
            "raw_data_policy": "outside_repo_only",
        },
        "synthetic_sleep_machinery": synthetic,
        "real_sleep_edf": real_data,
        "synthetic_sleep_machinery_passed": synthetic_passed,
        "gate_passed": bool(synthetic_passed and real_data["status"] == "parsed_local_sleep_edf"),
        "stop_reason": "M2 gate reached; real Sleep-EDF validation must pass before M3.",
    }
    _write_json(out / "m2_gate_report.json", gate)
    _write_report(out / "M2_EVIDENCE_REPORT.md", gate)
    return gate


def download_sleep_edf_subset(root: str | Path, *, n_pairs: int = 4) -> list[dict[str, str]]:
    root_path = Path(root)
    index = _sleep_edf_directory_index()
    psg_files = [name for name in index if name.endswith("-PSG.edf")]
    hyp_by_prefix = {name[:6]: name for name in index if name.endswith("-Hypnogram.edf")}
    pairs = []
    for psg in psg_files:
        hyp = hyp_by_prefix.get(psg[:6])
        if hyp is not None:
            pairs.append((psg, hyp))
        if len(pairs) >= n_pairs:
            break
    downloaded = []
    for psg, hyp in pairs:
        psg_path = _download_physionet_file(root_path, "sleep-cassette/" + psg)
        hyp_path = _download_physionet_file(root_path, "sleep-cassette/" + hyp)
        downloaded.append({"psg": str(psg_path), "hypnogram": str(hyp_path)})
    return downloaded


def make_synthetic_sleep_fixture(
    *,
    seed: int,
    n_subjects: int = 14,
    epochs_per_subject: int = 140,
    window: int = 64,
    channels: int = 4,
) -> SleepFixture:
    rng = np.random.default_rng(seed)
    t_axis = np.linspace(0.0, 1.0, window, dtype=np.float32)
    windows: list[np.ndarray] = []
    stages: list[int] = []
    transitions: list[int] = []
    subjects: list[int] = []
    sessions: list[int] = []
    datasets: list[int] = []
    sites: list[int] = []
    nuisance: list[list[float]] = []
    for subject in range(n_subjects):
        dataset = int(subject >= n_subjects // 2)
        site = subject % 3
        stage = int(rng.integers(0, 2))
        raw_stages = []
        raw_windows = []
        raw_events = []
        for epoch in range(epochs_per_subject + 1):
            circadian = np.sin(2.0 * np.pi * epoch / epochs_per_subject)
            transition_pressure = 0.65 * np.sin(2.0 * np.pi * epoch / 32.0) + rng.normal(0.0, 0.4)
            change_prob = _sigmoid(-2.1 + 0.8 * circadian + 1.35 * transition_pressure)
            event = int(rng.random() < change_prob)
            raw_stages.append(stage)
            raw_events.append(event)
            signal = rng.normal(0.0, 0.18, size=(window, channels)).astype(np.float32)
            delta = (4 - stage) / 4.0
            alpha = stage / 4.0
            signal[:, 0] += (0.8 * delta + 0.15 * transition_pressure) * np.sin(2.0 * np.pi * 2.0 * t_axis)
            signal[:, 1] += (0.65 * alpha + 0.18 * transition_pressure) * np.sin(2.0 * np.pi * 10.0 * t_axis)
            signal[:, 2] += 0.45 * np.sin(2.0 * np.pi * 6.0 * t_axis + float(site))
            signal[:, 3] = 0.55 * signal[:, 0] - 0.25 * signal[:, 1] + rng.normal(0.0, 0.08, size=window)
            raw_windows.append(signal)
            if event:
                stage = int(np.clip(stage + rng.choice([-1, 1]), 0, 4))
        recent_changes = [0]
        for epoch in range(epochs_per_subject):
            next_transition = int(raw_events[epoch])
            recent = float(np.mean(recent_changes[-10:]))
            windows.append(raw_windows[epoch])
            stages.append(raw_stages[epoch])
            transitions.append(next_transition)
            subjects.append(subject)
            sessions.append(dataset)
            datasets.append(dataset)
            sites.append(site)
            nuisance.append([1.0, np.sin(2.0 * np.pi * epoch / epochs_per_subject), np.cos(2.0 * np.pi * epoch / epochs_per_subject), recent, raw_stages[epoch] / 4.0])
            recent_changes.append(next_transition)
    return SleepFixture(
        windows=np.asarray(windows, dtype=np.float32),
        stages=np.asarray(stages, dtype=np.int64),
        transition=np.asarray(transitions, dtype=np.int64),
        subject=np.asarray(subjects, dtype=np.int64),
        session=np.asarray(sessions, dtype=np.int64),
        dataset=np.asarray(datasets, dtype=np.int64),
        site=np.asarray(sites, dtype=np.int64),
        nuisance=np.asarray(nuisance, dtype=np.float32),
    )


def sleep_edf_manifest_audit(root: str | Path | None) -> dict[str, Any]:
    if root is None:
        return {
            "status": "not_run_no_local_sleep_edf_root",
            "required_pattern": "*-PSG.edf with matching *-Hypnogram.edf",
            "parser": "requires local EDF reader such as mne; raw data must stay outside repo",
        }
    root_path = Path(root)
    pairs = _local_sleep_edf_pairs(root_path)
    if len(pairs) < 3:
        return {
            "status": "local_manifest_insufficient",
            "root": str(root_path),
            "pairs": len(pairs),
            "minimum_pairs": 3,
            "parsed_local_sleep_edf": False,
        }
    try:
        used_pairs = min(8, len(pairs))
        fixture = load_sleep_edf_fixture(root_path, max_pairs=used_pairs)
        payload = {
            "transition_hazard": _run_fixture(_as_transition_fixture(fixture), seed=0),
            "held_out_channel_reconstruction": _held_out_channel_reconstruction(fixture, target_idx=1),
            "leakage_audit": _sleep_leakage_audit(label_standard="Sleep-EDF hypnogram labels; R&K/AASM mismatch must be recorded"),
        }
        failures = _real_sleep_edf_gate_failures(payload)
        passed = not failures
        return {
            "status": "parsed_local_sleep_edf" if passed else "parsed_local_sleep_edf_gate_failed",
            "root": str(root_path),
            "pairs": len(pairs),
            "used_pairs": used_pairs,
            "file_hashes": _pair_hashes(pairs[:used_pairs]),
            "payload": payload,
            "gate_failures": failures,
            "parsed_local_sleep_edf": True,
            "real_sleep_edf_gate_passed": passed,
        }
    except Exception as exc:  # noqa: BLE001 - parser failures are evidence.
        return {
            "status": "local_sleep_edf_parse_failed",
            "root": str(root_path),
            "pairs": len(pairs),
            "error": str(exc),
            "parsed_local_sleep_edf": False,
        }


def fetch_sleep_edf_records_index() -> list[str]:
    with urlopen(SLEEP_EDF_BASE_URL + "RECORDS", timeout=20) as response:
        return response.read().decode("utf-8").splitlines()


def load_sleep_edf_fixture(root: str | Path, *, max_pairs: int | None = 4) -> SleepFixture:
    pairs = _local_sleep_edf_pairs(Path(root))
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    pair_metadata = _sleep_edf_pair_metadata(pairs)
    subject_codes = {subject: idx for idx, subject in enumerate(dict.fromkeys(row["subject_id"] for row in pair_metadata))}
    session_codes = {
        session: idx
        for idx, session in enumerate(dict.fromkeys(row["subject_session_id"] for row in pair_metadata))
    }
    dataset_codes = {dataset: idx for idx, dataset in enumerate(dict.fromkeys(row["dataset_id"] for row in pair_metadata))}
    windows: list[np.ndarray] = []
    stages: list[int] = []
    transitions: list[int] = []
    subjects: list[int] = []
    sessions: list[int] = []
    datasets: list[int] = []
    sites: list[int] = []
    nuisance: list[list[float]] = []
    for metadata, (psg_path, hyp_path) in zip(pair_metadata, pairs, strict=True):
        psg = _read_edf_signals(psg_path, preferred_labels=("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"))
        hyp = _read_sleep_edf_hypnogram(hyp_path)
        labels = _stage_per_record(hyp, n_records=psg["signals"].shape[0], record_seconds=float(psg["record_duration"]))
        valid = labels >= 0
        signals = psg["signals"][valid]
        labels = labels[valid]
        if len(labels) < 4:
            continue
        recent_changes = [0]
        for epoch in range(len(labels) - 1):
            transition = int(labels[epoch + 1] != labels[epoch])
            windows.append(signals[epoch])
            stages.append(int(labels[epoch]))
            transitions.append(transition)
            subjects.append(subject_codes[metadata["subject_id"]])
            sessions.append(session_codes[metadata["subject_session_id"]])
            datasets.append(dataset_codes[metadata["dataset_id"]])
            sites.append(dataset_codes[metadata["dataset_id"]])
            recent = float(np.mean(recent_changes[-10:]))
            nuisance.append([1.0, np.sin(2.0 * np.pi * epoch / max(1, len(labels))), np.cos(2.0 * np.pi * epoch / max(1, len(labels))), recent, labels[epoch] / 5.0])
            recent_changes.append(transition)
    if not windows:
        raise ValueError("no usable Sleep-EDF windows parsed")
    return SleepFixture(
        windows=np.asarray(windows, dtype=np.float32),
        stages=np.asarray(stages, dtype=np.int64),
        transition=np.asarray(transitions, dtype=np.int64),
        subject=np.asarray(subjects, dtype=np.int64),
        session=np.asarray(sessions, dtype=np.int64),
        dataset=np.asarray(datasets, dtype=np.int64),
        site=np.asarray(sites, dtype=np.int64),
        nuisance=np.asarray(nuisance, dtype=np.float32),
    )


def _local_sleep_edf_pairs(root: Path) -> list[tuple[Path, Path]]:
    psg = sorted(root.rglob("*-PSG.edf"))
    hyp_by_prefix = {path.name[:6]: path for path in root.rglob("*-Hypnogram.edf")}
    return [(path, hyp_by_prefix[path.name[:6]]) for path in psg if path.name[:6] in hyp_by_prefix]


def _sleep_edf_record_metadata(path: str | Path) -> dict[str, Any]:
    name = Path(path).name
    record_id = name.removesuffix("-PSG.edf").removesuffix("-Hypnogram.edf")
    match = re.match(r"^(?P<study>SC)4(?P<subject>\d{2})(?P<night>\d)[A-Z]\d$", record_id)
    dataset_id = "sleep-cassette"
    if match is None:
        match = re.match(r"^(?P<study>ST)7(?P<subject>\d{2})(?P<night>\d)[A-Z]\d$", record_id)
        dataset_id = "sleep-telemetry"
    if match is None:
        raise ValueError(f"Sleep-EDF filename is not recognized: {name}")
    study = match.group("study")
    subject_id = f"{study}-{match.group('subject')}"
    night = int(match.group("night"))
    session_id = f"night-{night}"
    return {
        "record_id": record_id,
        "study_code": study,
        "dataset_id": dataset_id,
        "subject_id": subject_id,
        "night": night,
        "session_id": session_id,
        "subject_session_id": f"{subject_id}-{session_id}",
    }


def _sleep_edf_pair_metadata(pairs: list[tuple[Path, Path]]) -> list[dict[str, Any]]:
    rows = []
    for psg, hyp in pairs:
        psg_metadata = _sleep_edf_record_metadata(psg)
        hyp_metadata = _sleep_edf_record_metadata(hyp)
        if psg_metadata["subject_session_id"] != hyp_metadata["subject_session_id"]:
            raise ValueError(f"PSG/hypnogram Sleep-EDF pair mismatch: {psg.name} vs {hyp.name}")
        rows.append(psg_metadata)
    return rows


def _sleep_edf_directory_index() -> list[str]:
    with urlopen(SLEEP_EDF_BASE_URL + "sleep-cassette/", timeout=20) as response:
        html = response.read().decode("utf-8", errors="ignore")
    return re.findall(r'href="([^"]+\.edf)"', html)


def _download_physionet_file(root: Path, rel_path: str) -> Path:
    target = root / rel_path
    remote_size = _remote_size(SLEEP_EDF_BASE_URL + rel_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    attempts = 0
    while attempts < 5:
        local_size = target.stat().st_size if target.exists() else 0
        if remote_size is not None and local_size == remote_size:
            return target
        headers = {"Range": f"bytes={local_size}-"} if local_size and remote_size and local_size < remote_size else {}
        mode = "ab" if headers else "wb"
        try:
            with urlopen(Request(SLEEP_EDF_BASE_URL + rel_path, headers=headers), timeout=120) as response, target.open(mode) as handle:
                if headers and getattr(response, "status", 200) != 206:
                    handle.seek(0)
                    handle.truncate()
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        except TimeoutError:
            attempts += 1
            continue
        attempts += 1
    if remote_size is not None and target.stat().st_size != remote_size:
        raise ValueError(f"incomplete download for {rel_path}: {target.stat().st_size} != {remote_size}")
    return target


def _remote_size(url: str) -> int | None:
    try:
        with urlopen(Request(url, method="HEAD"), timeout=20) as response:
            value = response.headers.get("Content-Length")
            return int(value) if value else None
    except Exception:
        return None


def _read_edf_signals(path: Path, *, preferred_labels: tuple[str, ...]) -> dict[str, Any]:
    with path.open("rb") as handle:
        fixed = handle.read(256)
        header_bytes = int(fixed[184:192].decode("ascii", errors="ignore").strip())
        n_records = int(fixed[236:244].decode("ascii", errors="ignore").strip())
        record_duration = float(fixed[244:252].decode("ascii", errors="ignore").strip())
        n_signals = int(fixed[252:256].decode("ascii", errors="ignore").strip())
        header = _read_edf_signal_header(handle, n_signals)
        handle.seek(header_bytes)
        data = np.frombuffer(handle.read(), dtype="<i2")
    samples = np.asarray([int(value) for value in header["samples"]], dtype=np.int64)
    wanted = [header["label"].index(label) for label in preferred_labels if label in header["label"]]
    if len(wanted) < 3:
        raise ValueError(f"missing required Sleep-EDF channels in {path}")
    total_per_record = int(np.sum(samples))
    if data.size < n_records * total_per_record:
        n_records = data.size // total_per_record
    records = []
    cursor = 0
    for _ in range(n_records):
        pieces = []
        for sig_idx in range(n_signals):
            raw = data[cursor : cursor + samples[sig_idx]]
            cursor += int(samples[sig_idx])
            if sig_idx in wanted:
                pieces.append(_resample_1d(_digital_to_physical(raw, header, sig_idx), 64))
        records.append(np.stack(pieces, axis=1))
    return {"signals": np.asarray(records, dtype=np.float32), "record_duration": record_duration, "labels": [header["label"][idx] for idx in wanted]}


def _read_edf_signal_header(handle: Any, n_signals: int) -> dict[str, list[str]]:
    fields = {}
    for name, width in (
        ("label", 16),
        ("transducer", 80),
        ("physdim", 8),
        ("physmin", 8),
        ("physmax", 8),
        ("digmin", 8),
        ("digmax", 8),
        ("prefilter", 80),
        ("samples", 8),
        ("reserved", 32),
    ):
        fields[name] = [handle.read(width).decode("latin1", errors="ignore").strip() for _ in range(n_signals)]
    return fields


def _digital_to_physical(raw: np.ndarray, header: dict[str, list[str]], sig_idx: int) -> np.ndarray:
    digmin = float(header["digmin"][sig_idx])
    digmax = float(header["digmax"][sig_idx])
    physmin = float(header["physmin"][sig_idx])
    physmax = float(header["physmax"][sig_idx])
    return physmin + (raw.astype(np.float32) - digmin) * (physmax - physmin) / (digmax - digmin)


def _resample_1d(values: np.ndarray, n: int) -> np.ndarray:
    if values.size == n:
        return values.astype(np.float32)
    xp = np.linspace(0.0, 1.0, values.size)
    x = np.linspace(0.0, 1.0, n)
    return np.interp(x, xp, values).astype(np.float32)


def _read_sleep_edf_hypnogram(path: Path) -> list[tuple[float, float, int]]:
    with path.open("rb") as handle:
        fixed = handle.read(256)
        n_signals = int(fixed[252:256].decode("ascii", errors="ignore").strip())
        _read_edf_signal_header(handle, n_signals)
        text = handle.read().decode("latin1", errors="ignore")
    rows = []
    for chunk in text.split("\x00"):
        fields = [part for part in chunk.split("\x14") if part]
        if len(fields) < 2 or not fields[0].startswith("+"):
            continue
        onset_duration = fields[0].split("\x15")
        if len(onset_duration) < 2:
            continue
        label = fields[1]
        stage = _stage_label_to_int(label)
        if stage is None:
            continue
        try:
            rows.append((float(onset_duration[0].lstrip("+")), float(onset_duration[1]), stage))
        except ValueError:
            continue
    return rows


def _stage_label_to_int(label: str) -> int | None:
    mapping = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 4,
        "Sleep stage R": 5,
    }
    return mapping.get(label)


def _stage_per_record(hypnogram: list[tuple[float, float, int]], *, n_records: int, record_seconds: float) -> np.ndarray:
    labels = np.full(n_records, -1, dtype=np.int64)
    for onset, duration, stage in hypnogram:
        start = max(0, int(np.floor(onset / record_seconds)))
        stop = min(n_records, int(np.ceil((onset + duration) / record_seconds)))
        labels[start:stop] = stage
    return labels


def _pair_hashes(pairs: list[tuple[Path, Path]]) -> list[dict[str, Any]]:
    return [
        {
            **metadata,
            "psg": path.name,
            "psg_sha256": _sha256(path),
            "hypnogram": hyp.name,
            "hypnogram_sha256": _sha256(hyp),
        }
        for metadata, (path, hyp) in zip(_sleep_edf_pair_metadata(pairs), pairs, strict=True)
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _as_transition_fixture(fixture: SleepFixture) -> TransitionFixture:
    return TransitionFixture(
        windows=fixture.windows,
        nuisance=fixture.nuisance,
        y=fixture.transition,
        patient=fixture.subject,
        site=fixture.site,
        time_bucket=fixture.stages,
        session=fixture.session,
    )


def _held_out_channel_reconstruction(fixture: SleepFixture, *, target_idx: int | None = None) -> dict[str, float]:
    target_idx = fixture.windows.shape[-1] - 1 if target_idx is None else int(target_idx)
    source_idx = [idx for idx in range(fixture.windows.shape[-1]) if idx != target_idx]
    x = fixture.windows[:, :, source_idx]
    y = fixture.windows[:, :, target_idx : target_idx + 1]
    subjects = np.unique(fixture.subject)
    if len(subjects) < 2:
        return {"mse": float("nan"), "train_mean_mse": float("nan"), "r2": 0.0, "status": "underpowered_single_subject"}
    train_subjects = set(subjects[:-1])
    train = np.asarray([subject in train_subjects for subject in fixture.subject], dtype=bool)
    model = NumpyRidgeBaseline(alpha=1e-2)
    model.fit(x[train].reshape(-1, 3), y[train].reshape(-1, 1))
    pred = model.predict(x[~train].reshape(-1, 3)).reshape(y[~train].shape)
    mean = np.mean(y[train], axis=(0, 1), keepdims=True)
    mean_pred = np.broadcast_to(mean, y[~train].shape)
    return {
        "mse": mse(y[~train], pred),
        "train_mean_mse": mse(y[~train], mean_pred),
        "r2": r2_score(y[~train], pred),
    }


def _cross_dataset_transfer(fixture: SleepFixture, *, seed: int) -> dict[str, Any]:
    train = fixture.dataset == 0
    test = fixture.dataset == 1
    source = TransitionFixture(
        windows=fixture.windows[train],
        nuisance=fixture.nuisance[train],
        y=fixture.transition[train],
        patient=fixture.subject[train],
        site=fixture.site[train],
        time_bucket=fixture.stages[train],
        session=fixture.dataset[train],
    )
    target = TransitionFixture(
        windows=fixture.windows[test],
        nuisance=fixture.nuisance[test],
        y=fixture.transition[test],
        patient=fixture.subject[test],
        site=fixture.site[test],
        time_bucket=fixture.stages[test],
        session=fixture.dataset[test],
    )
    source_payload = _run_fixture(source, seed=seed + 17)
    target_payload = _run_fixture(target, seed=seed + 23)
    return {
        "dataset0_crossfit_rfs_bits": source_payload["logistic_full"]["rfs_bits"],
        "dataset1_crossfit_rfs_bits": target_payload["logistic_full"]["rfs_bits"],
        "exploratory_note": "synthetic split only; real cross-dataset transfer requires MASS/SHHS or another PSG corpus",
    }


def _sleep_leakage_audit(*, label_standard: str) -> dict[str, Any]:
    return {
        "split_unit": "subject",
        "per_recording_normalization": "blocked",
        "same_subject_nights": "must stay within one split",
        "window_overlap": "synthetic fixture uses epoch-level non-overlap",
        "label_standard": label_standard,
        "wake_trimming": "not applied silently",
    }


def _synthetic_sleep_passes(payload: dict[str, Any]) -> bool:
    hazard = payload["transition_hazard"]
    recon = payload["held_out_channel_reconstruction"]
    full = hazard["logistic_full"]
    shuffled = hazard["shuffled_target_control"]
    return bool(
        hazard["positive_events"] >= 80
        and hazard["event_patients"] >= 10
        and full["rfs_ci_low"] > 0.0
        and shuffled["rfs_bits"] < full["rfs_bits"] * 0.4
        and recon["mse"] < recon["train_mean_mse"] * 0.5
    )


def _real_sleep_edf_passes(payload: dict[str, Any]) -> bool:
    return not _real_sleep_edf_gate_failures(payload)


def _real_sleep_edf_gate_failures(payload: dict[str, Any]) -> list[str]:
    hazard = payload["transition_hazard"]
    recon = payload["held_out_channel_reconstruction"]
    full = hazard["logistic_full"]
    shuffled = hazard["shuffled_target_control"]
    failures = []
    if hazard["positive_events"] < 30:
        failures.append("underpowered_positive_events")
    if hazard["event_patients"] < 6:
        failures.append("underpowered_event_patients")
    if full["rfs_ci_low"] <= 0.0:
        failures.append("primary_rfs_ci_includes_zero")
    if shuffled["rfs_bits"] >= full["rfs_bits"] * 0.5:
        failures.append("shuffled_target_control_too_close")
    if not np.isfinite(recon["mse"]) or recon["mse"] >= recon["train_mean_mse"]:
        failures.append("held_out_channel_reconstruction_not_better_than_mean")
    for key in ("patient", "site"):
        probe = hazard["nuisance_probes"][key]
        if probe["accuracy"] > probe["chance"] + 0.20:
            failures.append(f"nuisance_probe_{key}_above_threshold")
    return failures


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_report(path: Path, gate: dict[str, Any]) -> None:
    synthetic = gate["synthetic_sleep_machinery"]
    hazard = synthetic["transition_hazard"]
    recon = synthetic["held_out_channel_reconstruction"]
    lines = [
        "# Kahlus Forecastability Trial 0 - M2 Evidence Report",
        "",
        f"Gate passed: `{gate['gate_passed']}`",
        f"Synthetic sleep machinery passed: `{gate['synthetic_sleep_machinery_passed']}`",
        f"Real Sleep-EDF status: `{gate['real_sleep_edf']['status']}`",
        "",
        "## Synthetic Sleep Transition Hazard",
        "",
        f"- rows/events/event-patients: `{hazard['n']}` / `{hazard['positive_events']}` / `{hazard['event_patients']}`",
        f"- RFS bits: `{hazard['logistic_full']['rfs_bits']:.6f}` CI `[ {hazard['logistic_full']['rfs_ci_low']:.6f}, {hazard['logistic_full']['rfs_ci_high']:.6f} ]`",
        f"- shuffled-target RFS bits: `{hazard['shuffled_target_control']['rfs_bits']:.6f}`",
        f"- time-shift RFS bits: `{hazard['time_shift_control']['rfs_bits']:.6f}`",
        "",
        "## Held-Out Channel Reconstruction",
        "",
        f"- ridge MSE: `{recon['mse']:.6f}`",
        f"- train-mean MSE: `{recon['train_mean_mse']:.6f}`",
        f"- R2: `{recon['r2']:.6f}`",
        "",
    ]
    real = gate["real_sleep_edf"]
    if "payload" in real:
        real_hazard = real["payload"]["transition_hazard"]
        real_recon = real["payload"]["held_out_channel_reconstruction"]
        lines.extend(
            [
                "## Real Sleep-EDF Subset",
                "",
                f"- pairs used: `{real['used_pairs']}`",
                f"- gate failures: `{', '.join(real.get('gate_failures', [])) or 'none'}`",
                f"- rows/events/event-patients: `{real_hazard['n']}` / `{real_hazard['positive_events']}` / `{real_hazard['event_patients']}`",
                f"- RFS bits: `{real_hazard['logistic_full']['rfs_bits']:.6f}` CI `[ {real_hazard['logistic_full']['rfs_ci_low']:.6f}, {real_hazard['logistic_full']['rfs_ci_high']:.6f} ]`",
                f"- shuffled-target RFS bits: `{real_hazard['shuffled_target_control']['rfs_bits']:.6f}`",
                f"- channel reconstruction MSE: `{real_recon['mse']:.6f}` vs mean `{real_recon['train_mean_mse']:.6f}`",
                "",
            ]
        )
    if not gate["gate_passed"]:
        lines.append("M2 does not pass until local real Sleep-EDF validation is parsed and passes. Raw PSG files must remain outside the repo.")
    else:
        lines.append("M2 passed on a tiny real Sleep-EDF subset. This is a machinery gate, not a publication-scale sleep result.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
