#!/usr/bin/env python3
"""Download OpenNeuro ds005620 (propofol sedation EEG) into local datasets root."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_ROOT = Path("/Users/aayu/datasets/kahlus_multidataset_public/ds005620")
DATASET_ID = "ds005620"
S3_URI = f"s3://openneuro.org/{DATASET_ID}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_via_aws(target: Path) -> None:
    if shutil.which("aws") is None:
        raise RuntimeError("aws CLI not found; install AWS CLI or download ds005620 manually")
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["aws", "s3", "sync", "--no-sign-request", S3_URI, str(target)],
        check=True,
    )


def _subject_count(root: Path) -> int:
    participants = root / "participants.tsv"
    if not participants.is_file():
        raise FileNotFoundError(f"missing {participants}")
    lines = [line for line in participants.read_text(encoding="utf-8").splitlines() if line.strip()]
    return max(0, len(lines) - 1)


def write_fingerprint(root: Path) -> dict[str, object]:
    participants = root / "participants.tsv"
    vhdr_files = sorted(p.name for p in root.rglob("*.vhdr"))
    payload = {
        "dataset_id": DATASET_ID,
        "participants_sha256": _sha256(participants),
        "n_subjects": _subject_count(root),
        "n_vhdr_files": len(vhdr_files),
        "vhdr_files": vhdr_files[:50],
        "vhdr_truncated": len(vhdr_files) > 50,
    }
    out = root / "dataset_fingerprint.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch OpenNeuro ds005620 propofol EEG dataset.")
    parser.add_argument("--target", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--skip-download", action="store_true", help="Only verify + write fingerprint.")
    args = parser.parse_args()
    target = args.target
    if not args.skip_download:
        print(f"downloading {DATASET_ID} to {target} ...")
        _download_via_aws(target)
    if not (target / "participants.tsv").is_file():
        print(f"error: {target} does not look like ds005620 (missing participants.tsv)", file=sys.stderr)
        return 1
    n_subjects = _subject_count(target)
    if n_subjects != 21:
        print(f"warning: expected 21 subjects, found {n_subjects}", file=sys.stderr)
    fingerprint = write_fingerprint(target)
    print(json.dumps(fingerprint, indent=2))
    print(f"fingerprint: {target / 'dataset_fingerprint.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
