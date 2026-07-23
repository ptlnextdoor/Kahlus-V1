#!/usr/bin/env python3
"""Download OpenNeuro ds005620 (propofol sedation EEG) into local datasets root."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_ROOT = Path("/Users/aayu/datasets/kahlus_multidataset_public/ds005620")
DATASET_ID = "ds005620"
S3_URI = f"s3://openneuro.org/{DATASET_ID}"
PARTICIPANTS_URL = (
    "https://openneuro.org/crn/datasets/ds005620/snapshots/1.0.0/files/participants.tsv"
)
REST_TASK_GLOBS = (
    "*task-awake_acq-EC*",
    "*task-sed_acq-rest*",
    "*task-sed2_acq-rest*",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_openneuro_py() -> None:
    try:
        import openneuro  # noqa: F401
    except ImportError:
        if shutil.which("uv") is not None:
            subprocess.run(["uv", "pip", "install", "--python", sys.executable, "openneuro-py", "-q"], check=True)
        else:
            subprocess.run([sys.executable, "-m", "pip", "install", "openneuro-py", "-q"], check=True)


def _discover_subject_dirs() -> list[str]:
    _ensure_openneuro_py()
    from openneuro import download

    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp)
        download(dataset=DATASET_ID, target_dir=str(probe), include=["sub-*/eeg/*.json"])
        subjects = sorted(
            {
                path.parts[-3]
                for path in probe.rglob("*.json")
                if len(path.parts) >= 3 and path.parts[-3].startswith("sub-")
            }
        )
    return subjects


def _rest_include_for_subject(subject: str) -> list[str]:
    return [f"{subject}/eeg/{pattern}" for pattern in REST_TASK_GLOBS]


def _download_via_openneuro(target: Path, *, max_subjects: int) -> None:
    _ensure_openneuro_py()
    from openneuro import download

    target.mkdir(parents=True, exist_ok=True)
    subjects = _discover_subject_dirs()[:max_subjects]
    if len(subjects) < 1:
        raise RuntimeError("could not discover ds005620 subject directories")
    download(dataset=DATASET_ID, target_dir=str(target), include=["participants.tsv", "dataset_description.json"])
    for subject in subjects:
        download(dataset=DATASET_ID, target_dir=str(target), include=_rest_include_for_subject(subject))


def _download_via_aws(target: Path) -> None:
    if shutil.which("aws") is None:
        raise RuntimeError("aws CLI not found")
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["aws", "s3", "sync", "--no-sign-request", S3_URI, str(target)],
        check=True,
    )


def download_dataset(target: Path, *, max_subjects: int) -> None:
    errors: list[str] = []
    try:
        _download_via_openneuro(target, max_subjects=max_subjects)
        return
    except Exception as exc:  # noqa: BLE001 - try aws fallback
        errors.append(f"_download_via_openneuro: {exc}")
    try:
        _download_via_aws(target)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"_download_via_aws: {exc}")
        raise RuntimeError("all download methods failed: " + "; ".join(errors)) from exc


def _subject_count(root: Path) -> int:
    participants = root / "participants.tsv"
    if participants.is_file():
        lines = [line for line in participants.read_text(encoding="utf-8").splitlines() if line.strip()]
        return max(0, len(lines) - 1)
    return len({p.parts[-3] for p in root.rglob("*_eeg.vhdr") if p.parts[-3].startswith("sub-")})


def write_fingerprint(root: Path) -> dict[str, object]:
    participants = root / "participants.tsv"
    vhdr_files = sorted(
        p.name for p in root.rglob("*.vhdr") if "tms" not in p.as_posix().lower()
    )
    payload = {
        "dataset_id": DATASET_ID,
        "participants_sha256": _sha256(participants) if participants.is_file() else None,
        "n_subjects": _subject_count(root),
        "n_vhdr_files": len(vhdr_files),
        "vhdr_files": vhdr_files[:50],
        "vhdr_truncated": len(vhdr_files) > 50,
        "rest_only_no_tms": True,
        "bids_layout": "sub-*/eeg/ (no ses- prefix in paths)",
    }
    out = root / "dataset_fingerprint.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch OpenNeuro ds005620 propofol EEG dataset.")
    parser.add_argument("--target", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=21,
        help="Download at most this many subjects (rest EEG only, excludes TMS runs).",
    )
    parser.add_argument("--skip-download", action="store_true", help="Only verify + write fingerprint.")
    args = parser.parse_args()
    target = args.target
    if not args.skip_download:
        print(
            f"downloading {DATASET_ID} to {target} "
            f"(max_subjects={args.max_subjects}, rest-only, no TMS) ..."
        )
        download_dataset(target, max_subjects=args.max_subjects)
    if not (target / "participants.tsv").is_file() and not list(target.rglob("*.vhdr")):
        print(f"error: {target} does not look like ds005620", file=sys.stderr)
        return 1
    fingerprint = write_fingerprint(target)
    print(json.dumps(fingerprint, indent=2))
    print(f"fingerprint: {target / 'dataset_fingerprint.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
