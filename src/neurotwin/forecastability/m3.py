from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

from neurotwin.forecastability.m1 import TransitionFixture, _run_fixture
from neurotwin.forecastability.m2 import _read_edf_signals, _remote_size, _sha256


CHBMIT_BASE_URL = "https://physionet.org/files/chbmit/1.0.0/"
TUSZ_AUDIT_URL = "https://isip.piconepress.com/projects/nedc/html/tuh_eeg/"
PRIMARY_HORIZON_SECONDS = 300


@dataclass(frozen=True)
class ChbmitRecording:
    subject: str
    edf_path: Path
    summary_path: Path
    seizures: tuple[tuple[int, int], ...]


def run_m3_gate(out_dir: str | Path, *, seed: int = 0, chbmit_root: str | Path | None = None) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    source_audit = {
        "chb_mit": chbmit_source_audit(chbmit_root),
        "external_scalp_corpus": tusz_source_audit(),
        "primary_horizon_seconds": PRIMARY_HORIZON_SECONDS,
        "claim_scope": "seizure_forecastability_screening_only_not_clinical_prediction",
    }
    if chbmit_root is None:
        chb_payload = {"status": "not_run_no_local_chbmit_root"}
    else:
        chb_payload = _run_chbmit_smoke(Path(chbmit_root), seed=seed)
    failures = _m3_gate_failures(chb_payload, source_audit)
    gate = {
        "milestone": "M3",
        "source_audit": source_audit,
        "chb_mit_development": chb_payload,
        "gate_failures": failures,
        "gate_passed": not failures,
        "forecastability_class": _forecastability_class(chb_payload, failures),
        "stop_reason": "M3 gate reached; do not proceed to M4 until this report is reviewed.",
    }
    _write_json(out / "m3_gate_report.json", gate)
    _write_report(out / "M3_EVIDENCE_REPORT.md", gate)
    return gate


def download_chbmit_subset(root: str | Path, *, subjects: tuple[str, ...] = ("chb01", "chb02", "chb03")) -> list[dict[str, str]]:
    root_path = Path(root)
    downloaded = []
    for subject in subjects:
        summary_rel = f"{subject}/{subject}-summary.txt"
        summary_path = _download_chbmit_file(root_path, summary_rel)
        seizures = parse_chbmit_summary(summary_path.read_text(encoding="utf-8", errors="ignore"))
        first_seizure_file = next(name for name, rows in seizures.items() if rows)
        edf_path = _download_chbmit_file(root_path, f"{subject}/{first_seizure_file}")
        downloaded.append({"subject": subject, "summary": str(summary_path), "edf": str(edf_path)})
    return downloaded


def _download_chbmit_file(root: Path, rel_path: str) -> Path:
    target = root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    url = CHBMIT_BASE_URL + rel_path
    remote_size = _remote_size(url)
    if remote_size is not None and target.exists() and target.stat().st_size == remote_size:
        return target
    local_size = target.stat().st_size if target.exists() else 0
    headers = {"Range": f"bytes={local_size}-"} if local_size and remote_size and local_size < remote_size else {}
    mode = "ab" if headers else "wb"
    with urlopen(Request(url, headers=headers), timeout=120) as response, target.open(mode) as handle:
        if headers and getattr(response, "status", 200) != 206:
            handle.seek(0)
            handle.truncate()
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    if remote_size is not None and target.stat().st_size != remote_size:
        raise ValueError(f"incomplete CHB-MIT download for {rel_path}: {target.stat().st_size} != {remote_size}")
    return target


def chbmit_source_audit(root: str | Path | None) -> dict[str, Any]:
    records = fetch_chbmit_seizure_records()
    if root is None:
        return {"status": "not_run_no_local_chbmit_root", "official_records_with_seizures_count": len(records), "url": CHBMIT_BASE_URL}
    recordings = _local_chbmit_recordings(Path(root))
    return {
        "status": "local_manifest",
        "root": str(root),
        "official_records_with_seizures_count": len(records),
        "local_seizure_recordings": len(recordings),
        "local_event_patients": len({recording.subject for recording in recordings}),
        "url": CHBMIT_BASE_URL,
    }


def tusz_source_audit() -> dict[str, Any]:
    return {
        "status": "not_run_requires_external_tusz_access",
        "url": TUSZ_AUDIT_URL,
        "reason": "TUSZ/TUH EEG access is not an unauthenticated PhysioNet file path in this environment.",
    }


def fetch_chbmit_seizure_records() -> list[str]:
    with urlopen(CHBMIT_BASE_URL + "RECORDS-WITH-SEIZURES", timeout=20) as response:
        return response.read().decode("utf-8").splitlines()


def parse_chbmit_summary(text: str) -> dict[str, tuple[tuple[int, int], ...]]:
    current: str | None = None
    starts: list[int] = []
    rows: dict[str, list[tuple[int, int]]] = {}
    for line in text.splitlines():
        if line.startswith("File Name:"):
            current = line.split(":", 1)[1].strip()
            rows.setdefault(current, [])
            starts = []
        elif "Seizure Start Time:" in line:
            starts.append(int(re.search(r"(\d+)", line).group(1)))  # type: ignore[union-attr]
        elif "Seizure End Time:" in line and current is not None and starts:
            rows[current].append((starts.pop(0), int(re.search(r"(\d+)", line).group(1))))  # type: ignore[union-attr]
    return {name: tuple(values) for name, values in rows.items()}


def _run_chbmit_smoke(root: Path, *, seed: int) -> dict[str, Any]:
    recordings = _local_chbmit_recordings(root)
    if len({recording.subject for recording in recordings}) < 2:
        return {"status": "underpowered_local_chbmit_subset", "recordings": len(recordings)}
    fixture = _load_chbmit_fixture(recordings)
    payload = _run_fixture(fixture, seed=seed)
    return {
        "status": "completed_chbmit_development_smoke",
        "recordings": len(recordings),
        "file_hashes": [
            {
                "subject": recording.subject,
                "edf": recording.edf_path.name,
                "edf_sha256": _sha256(recording.edf_path),
                "summary": recording.summary_path.name,
                "summary_sha256": _sha256(recording.summary_path),
                "seizures": list(recording.seizures),
            }
            for recording in recordings
        ],
        "primary_horizon_seconds": PRIMARY_HORIZON_SECONDS,
        "preictal_exclusion": "ictal and 300s postictal rows skipped",
        "metrics": payload,
    }


def _local_chbmit_recordings(root: Path) -> list[ChbmitRecording]:
    recordings = []
    for summary_path in sorted(root.rglob("*-summary.txt")):
        subject = summary_path.parent.name
        by_file = parse_chbmit_summary(summary_path.read_text(encoding="utf-8", errors="ignore"))
        for file_name, seizures in by_file.items():
            edf_path = summary_path.parent / file_name
            if seizures and edf_path.exists():
                recordings.append(ChbmitRecording(subject=subject, edf_path=edf_path, summary_path=summary_path, seizures=seizures))
    return recordings


def _load_chbmit_fixture(recordings: list[ChbmitRecording]) -> TransitionFixture:
    windows: list[np.ndarray] = []
    labels: list[int] = []
    subjects: list[int] = []
    sites: list[int] = []
    time_buckets: list[int] = []
    sessions: list[int] = []
    nuisance: list[list[float]] = []
    subject_ids = {subject: idx for idx, subject in enumerate(sorted({recording.subject for recording in recordings}))}
    for session_idx, recording in enumerate(recordings):
        edf = _read_edf_signals(recording.edf_path, preferred_labels=("FP1-F7", "F7-T7", "T7-P7", "P7-O1"))
        signal = edf["signals"]
        record_seconds = float(edf["record_duration"])
        rows = _preictal_rows(recording.seizures, n_rows=signal.shape[0], record_seconds=record_seconds)
        recent = [0]
        for row_idx, label in rows:
            epoch = row_idx * record_seconds
            windows.append(signal[row_idx])
            labels.append(label)
            subjects.append(subject_ids[recording.subject])
            sites.append(0)
            time_buckets.append(int(epoch // 300) % 12)
            sessions.append(session_idx)
            nuisance.append([1.0, np.sin(2.0 * np.pi * epoch / 86400.0), np.cos(2.0 * np.pi * epoch / 86400.0), float(np.mean(recent[-300:])), (epoch % 3600.0) / 3600.0])
            recent.append(label)
    return TransitionFixture(
        windows=np.asarray(windows, dtype=np.float32),
        nuisance=np.asarray(nuisance, dtype=np.float32),
        y=np.asarray(labels, dtype=np.int64),
        patient=np.asarray(subjects, dtype=np.int64),
        site=np.asarray(sites, dtype=np.int64),
        time_bucket=np.asarray(time_buckets, dtype=np.int64),
        session=np.asarray(sessions, dtype=np.int64),
    )


def _preictal_rows(seizures: tuple[tuple[int, int], ...], *, n_rows: int, record_seconds: float) -> list[tuple[int, int]]:
    rows = []
    refractory = 300
    for row_idx in range(n_rows):
        t = row_idx * record_seconds
        if any(start <= t <= end + refractory for start, end in seizures):
            continue
        label = int(any(t < start and start - t <= PRIMARY_HORIZON_SECONDS for start, _ in seizures))
        rows.append((row_idx, label))
    return rows


def _m3_gate_failures(chb_payload: dict[str, Any], source_audit: dict[str, Any]) -> list[str]:
    failures = []
    if chb_payload.get("status") != "completed_chbmit_development_smoke":
        failures.append("chbmit_development_not_completed")
        return failures
    metrics = chb_payload["metrics"]
    full = metrics["logistic_full"]
    if metrics["event_patients"] < 8:
        failures.append("underpowered_event_patients")
    if metrics["positive_events"] < 100:
        failures.append("underpowered_positive_windows")
    if full["rfs_ci_low"] <= 0.0:
        failures.append("primary_rfs_ci_includes_zero")
    if metrics["shuffled_target_control"]["rfs_bits"] >= full["rfs_bits"] * 0.5:
        failures.append("shuffled_target_control_too_close")
    if source_audit["external_scalp_corpus"]["status"] != "completed_external_dataset":
        failures.append("external_dataset_held_out_not_run")
    return failures


def _forecastability_class(chb_payload: dict[str, Any], failures: list[str]) -> str:
    if any(failure.startswith("underpowered") for failure in failures):
        return "UNDERPOWERED"
    if "external_dataset_held_out_not_run" in failures:
        return "development_only_no_F3_claim"
    if chb_payload.get("status") == "completed_chbmit_development_smoke" and not failures:
        return "candidate_F3_requires_external_replication"
    return "not_assigned"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_report(path: Path, gate: dict[str, Any]) -> None:
    chb = gate["chb_mit_development"]
    lines = [
        "# Kahlus Forecastability Trial 0 - M3 Evidence Report",
        "",
        f"Gate passed: `{gate['gate_passed']}`",
        f"Forecastability class: `{gate['forecastability_class']}`",
        f"Gate failures: `{', '.join(gate['gate_failures']) if gate['gate_failures'] else 'none'}`",
        "",
        "## Source Audit",
        "",
        f"- CHB-MIT: `{gate['source_audit']['chb_mit']['status']}`",
        f"- external scalp corpus: `{gate['source_audit']['external_scalp_corpus']['status']}`",
        f"- primary horizon: `{PRIMARY_HORIZON_SECONDS}` seconds",
    ]
    if chb.get("status") == "completed_chbmit_development_smoke":
        metrics = chb["metrics"]
        lines.extend(
            [
                "",
                "## CHB-MIT Development Smoke",
                "",
                f"- recordings: `{chb['recordings']}`",
                f"- rows/events/event-patients: `{metrics['n']}` / `{metrics['positive_events']}` / `{metrics['event_patients']}`",
                f"- RFS bits: `{metrics['logistic_full']['rfs_bits']:.6f}` CI `[ {metrics['logistic_full']['rfs_ci_low']:.6f}, {metrics['logistic_full']['rfs_ci_high']:.6f} ]`",
                f"- GBM RFS bits: `{metrics['gbm_full']['rfs_bits']:.6f}`",
                f"- shuffled-target RFS bits: `{metrics['shuffled_target_control']['rfs_bits']:.6f}`",
                f"- time-shift RFS bits: `{metrics['time_shift_control']['rfs_bits']:.6f}`",
            ]
        )
    lines.append("")
    lines.append("M3 stops here. No clinical seizure prediction claim is permitted from this gate.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
