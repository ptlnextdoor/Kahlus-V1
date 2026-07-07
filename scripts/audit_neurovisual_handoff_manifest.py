#!/usr/bin/env python3
"""Audit a local NV-1 handoff manifest.

This verifies the handoff manifest, claim gate, execution boundaries, and input
artifact checksums. It does not check raw files, download datasets, execute
adapters, run baselines/models, or launch A100/cluster jobs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.repro import hash_file  # noqa: E402


REQUIRED_FALSE_EXECUTION_FLAGS: tuple[str, ...] = (
    "bulk_dataset_download",
    "a100_jobs_launched",
    "cluster_jobs_launched",
    "metadata_queries_executed",
    "adapters_implemented",
    "baselines_run",
    "models_run",
    "raw_file_existence_checked",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", required=True)
    args = parser.parse_args()

    payload = audit_handoff_manifest(Path(args.handoff))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


def audit_handoff_manifest(handoff_path: Path) -> dict[str, Any]:
    failures: list[str] = []
    verified_artifacts: list[str] = []
    handoff = _read_json(handoff_path, failures)

    if handoff:
        if handoff.get("schema") != "kahlus.nv1.handoff_manifest.v1":
            failures.append("bad_schema:handoff_manifest")
        if not handoff.get("passed"):
            failures.append("handoff_manifest_not_passed")
        if handoff.get("failures"):
            failures.append("handoff_manifest_has_failures")
        claim_gate = handoff.get("claim_gate", {})
        if not isinstance(claim_gate, dict):
            failures.append("missing_claim_gate")
        else:
            if claim_gate.get("schema") != "kahlus.nv1.claim_gate.v1":
                failures.append("bad_schema:claim_gate")
            if not claim_gate.get("passed"):
                failures.append("claim_gate_failed")
            if claim_gate.get("blocked_claims_found"):
                failures.append("blocked_claims_found")
        _check_false_flags(handoff.get("execution", {}), failures, "handoff_manifest")
        for artifact in handoff.get("input_artifacts", []):
            if not isinstance(artifact, dict):
                failures.append("invalid_input_artifact")
                continue
            artifact_name = str(artifact.get("path", "<unknown>"))
            artifact_path = Path(str(artifact.get("absolute_path", "")))
            if not artifact_path.exists():
                failures.append(f"missing_input_artifact:{artifact_name}")
                continue
            actual_hash = hash_file(artifact_path)
            actual_size = artifact_path.stat().st_size
            if artifact.get("sha256") != actual_hash:
                failures.append(f"checksum_mismatch:{artifact_name}")
            if artifact.get("size_bytes") != actual_size:
                failures.append(f"size_mismatch:{artifact_name}")
            verified_artifacts.append(artifact_name)

    return {
        "schema": "kahlus.nv1.handoff_manifest_audit.v1",
        "handoff_path": str(handoff_path),
        "passed": not failures,
        "failures": failures,
        "verified_artifacts": verified_artifacts,
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


def _read_json(path: Path, failures: list[str]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        failures.append(f"missing_handoff:{path}")
        return {}
    except json.JSONDecodeError:
        failures.append(f"invalid_json:{path}")
        return {}
    if not isinstance(payload, dict):
        failures.append(f"invalid_json_object:{path}")
        return {}
    return payload


def _check_false_flags(execution: Any, failures: list[str], label: str) -> None:
    if not isinstance(execution, dict):
        failures.append(f"missing_execution:{label}")
        return
    for flag_name in REQUIRED_FALSE_EXECUTION_FLAGS:
        if execution.get(flag_name) is not False:
            failures.append(f"execution_flag_not_false:{label}:{flag_name}")


if __name__ == "__main__":
    raise SystemExit(main())
