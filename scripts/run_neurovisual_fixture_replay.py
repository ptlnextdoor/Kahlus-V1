#!/usr/bin/env python3
"""Replay the NV-1 synthetic split fixture through the local split-audit CLI.

This is a local handoff smoke only. It rebuilds the registry package, audits the
generated synthetic split manifest, and writes compact evidence. It does not
check raw files, download datasets, execute adapters, run baselines/models, or
launch A100/cluster jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from _bootstrap import ensure_src_import_path

REPO_ROOT = ensure_src_import_path(__file__)

from neurotwin.repro import write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    registry_package_dir = out_dir / "registry_package"
    evidence_path = out_dir / "neurovisual_fixture_replay_evidence.json"

    build_result = _run_local_script(
        "scripts/build_neurovisual_dataset_registry.py",
        "--out",
        str(registry_package_dir),
    )
    split_result: subprocess.CompletedProcess[str] | None = None
    split_payload: dict[str, Any] = {}
    if build_result.returncode == 0:
        split_result = _run_local_script(
            "scripts/audit_neurovisual_local_split.py",
            "--manifest",
            str(registry_package_dir / "neurovisual_synthetic_split_manifest.json"),
            "--registry",
            str(registry_package_dir / "neurovisual_dataset_registry.json"),
        )
        split_payload = _json_object_from_stdout(split_result.stdout)

    payload = _evidence_payload(
        out_dir=out_dir,
        registry_package_dir=registry_package_dir,
        build_result=build_result,
        split_result=split_result,
        split_payload=split_payload,
    )
    write_json(evidence_path, payload)

    print(f"branch=nv1 out_dir={out_dir.resolve()}")
    print(f"registry_package={registry_package_dir}")
    print(f"evidence={evidence_path}")
    print(f"registry_build_returncode={payload['registry_build_returncode']}")
    print(f"split_cli_returncode={payload['split_cli_returncode']}")
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


def _json_object_from_stdout(stdout: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _evidence_payload(
    *,
    out_dir: Path,
    registry_package_dir: Path,
    build_result: subprocess.CompletedProcess[str],
    split_result: subprocess.CompletedProcess[str] | None,
    split_payload: dict[str, Any],
) -> dict[str, Any]:
    split_cli_returncode = split_result.returncode if split_result is not None else None
    split_audit_passed = bool(split_payload.get("passed")) if split_payload else False
    return {
        "schema": "kahlus.nv1.fixture_replay_smoke.v1",
        "scope": "local synthetic split fixture replay through public CLI",
        "passed": build_result.returncode == 0 and split_cli_returncode == 0 and split_audit_passed,
        "out_dir": str(out_dir),
        "registry_package_dir": str(registry_package_dir),
        "registry_path": str(registry_package_dir / "neurovisual_dataset_registry.json"),
        "split_manifest_path": str(registry_package_dir / "neurovisual_synthetic_split_manifest.json"),
        "split_audit_path": str(registry_package_dir / "neurovisual_synthetic_split_audit.json"),
        "registry_build_returncode": build_result.returncode,
        "split_cli_returncode": split_cli_returncode,
        "split_audit_passed": split_audit_passed,
        "split_counts": split_payload.get("split_counts", {}),
        "split_failures": split_payload.get("failures", []),
        "execution": {
            "bulk_dataset_download": False,
            "a100_jobs_launched": False,
            "cluster_jobs_launched": False,
            "metadata_queries_executed": False,
            "adapters_implemented": False,
            "baselines_run": False,
            "models_run": False,
            "raw_file_existence_checked": False,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
