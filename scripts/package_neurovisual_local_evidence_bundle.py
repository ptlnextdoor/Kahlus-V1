#!/usr/bin/env python3
"""Package the local NV-1 evidence gate outputs for downstream handoff.

This builds a small tar.gz containing generated metadata/evidence artifacts only.
It does not check raw files, download datasets, execute adapters, run
baselines/models, include checkpoints, or launch A100/cluster jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

from _bootstrap import ensure_src_import_path

REPO_ROOT = ensure_src_import_path(__file__)

from neurotwin.repro import hash_file, write_json  # noqa: E402


FORBIDDEN_SUFFIXES: tuple[str, ...] = (
    ".edf",
    ".fif",
    ".bdf",
    ".set",
    ".vhdr",
    ".eeg",
    ".ckpt",
    ".pt",
    ".pth",
    ".safetensors",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_dir = out_dir / "local_evidence_gate"
    archive_path = out_dir / "neurovisual_local_evidence_bundle.tar.gz"
    manifest_path = out_dir / "neurovisual_local_evidence_bundle_manifest.json"

    gate_result = _run_local_script("scripts/run_neurovisual_local_evidence_gate.py", "--out-dir", str(gate_dir))
    artifact_paths = _bundle_artifact_paths(gate_dir) if gate_result.returncode == 0 else []
    forbidden_paths = _forbidden_paths(artifact_paths)
    if not forbidden_paths:
        _write_archive(archive_path=archive_path, artifact_paths=artifact_paths, source_root=gate_dir)
    payload = _manifest_payload(
        out_dir=out_dir,
        gate_dir=gate_dir,
        archive_path=archive_path,
        gate_result=gate_result,
        artifact_paths=artifact_paths,
        forbidden_paths=forbidden_paths,
    )
    write_json(manifest_path, payload)

    print(f"branch=nv1 out_dir={out_dir.resolve()}")
    print(f"evidence_gate_dir={gate_dir}")
    print(f"archive={archive_path}")
    print(f"manifest={manifest_path}")
    print(f"passed={payload['passed']}")
    print("bulk_dataset_download=false")
    print("a100_jobs_launched=false")
    return 0 if payload["passed"] else 1


def _run_local_script(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


def _bundle_artifact_paths(gate_dir: Path) -> list[Path]:
    allowed_suffixes = {".json", ".md"}
    return sorted(
        path
        for path in gate_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed_suffixes
    )


def _forbidden_paths(paths: list[Path]) -> list[str]:
    return [str(path) for path in paths if path.suffix.lower() in FORBIDDEN_SUFFIXES]


def _write_archive(*, archive_path: Path, artifact_paths: list[Path], source_root: Path) -> None:
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in artifact_paths:
            relative_path = path.relative_to(source_root)
            archive.add(path, arcname=Path("neurovisual_local_evidence_bundle") / relative_path)


def _manifest_payload(
    *,
    out_dir: Path,
    gate_dir: Path,
    archive_path: Path,
    gate_result: subprocess.CompletedProcess[str],
    artifact_paths: list[Path],
    forbidden_paths: list[str],
) -> dict[str, Any]:
    archive_exists = archive_path.exists()
    artifacts = [
        {
            "path": str(path.relative_to(gate_dir)),
            "sha256": hash_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in artifact_paths
    ]
    passed = gate_result.returncode == 0 and archive_exists and not forbidden_paths
    return {
        "schema": "kahlus.nv1.local_evidence_bundle_manifest.v1",
        "scope": "local NV-1 evidence bundle package for downstream handoff",
        "passed": passed,
        "out_dir": str(out_dir),
        "evidence_gate_dir": str(gate_dir),
        "evidence_gate_returncode": gate_result.returncode,
        "archive": {
            "path": archive_path.name,
            "absolute_path": str(archive_path),
            "sha256": hash_file(archive_path) if archive_exists else None,
            "size_bytes": archive_path.stat().st_size if archive_exists else None,
        },
        "artifacts": artifacts,
        "forbidden_paths": forbidden_paths,
        "execution": {
            "bulk_dataset_download": False,
            "a100_jobs_launched": False,
            "cluster_jobs_launched": False,
            "metadata_queries_executed": False,
            "adapters_implemented": False,
            "baselines_run": False,
            "models_run": False,
            "raw_file_existence_checked": False,
            "raw_private_data_included": False,
            "checkpoints_included": False,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
