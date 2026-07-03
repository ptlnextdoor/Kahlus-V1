#!/usr/bin/env python3
"""Audit a local NV-1 neurovisual registry evidence bundle.

This is a CPU/local verification lane only. It does not query catalogs, download data,
launch A100, torchrun, or cluster jobs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.repro import hash_file  # noqa: E402


REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "neurovisual_dataset_registry.json",
    "neurovisual_registry_verification_summary.json",
    "neurovisual_adapter_plan.json",
    "neurovisual_metadata_query_plan.json",
    "neurovisual_local_manifest_schema.json",
    "neurovisual_split_audit_plan.json",
    "neurovisual_synthetic_split_manifest.json",
    "neurovisual_synthetic_split_audit.json",
    "neurovisual_registry_claim_gate.json",
    "neurovisual_dataset_registry.md",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    args = parser.parse_args()

    payload = audit_neurovisual_registry_bundle(Path(args.artifact_dir))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


def audit_neurovisual_registry_bundle(artifact_dir: Path) -> dict[str, Any]:
    failures: list[str] = []
    verified_artifacts: list[str] = []
    manifest = _read_json(artifact_dir / "neurovisual_registry_evidence_manifest.json", failures)
    registry = _read_json(artifact_dir / "neurovisual_dataset_registry.json", failures)
    summary = _read_json(artifact_dir / "neurovisual_registry_verification_summary.json", failures)
    adapter_plan = _read_json(artifact_dir / "neurovisual_adapter_plan.json", failures)
    query_plan = _read_json(artifact_dir / "neurovisual_metadata_query_plan.json", failures)
    local_manifest_schema = _read_json(artifact_dir / "neurovisual_local_manifest_schema.json", failures)
    split_audit_plan = _read_json(artifact_dir / "neurovisual_split_audit_plan.json", failures)
    synthetic_split_audit = _read_json(artifact_dir / "neurovisual_synthetic_split_audit.json", failures)
    claim_gate = _read_json(artifact_dir / "neurovisual_registry_claim_gate.json", failures)

    if manifest:
        if manifest.get("schema") != "kahlus.nv1.registry_evidence_manifest.v1":
            failures.append("bad_schema:neurovisual_registry_evidence_manifest.json")
        _check_false_flags(
            manifest.get("execution", {}),
            failures,
            "neurovisual_registry_evidence_manifest.json",
            ("bulk_dataset_download", "a100_jobs_launched", "cluster_jobs_launched"),
        )
        manifest_artifacts = {row.get("path"): row for row in manifest.get("artifacts", [])}
        for artifact_name in REQUIRED_ARTIFACTS:
            row = manifest_artifacts.get(artifact_name)
            artifact_path = artifact_dir / artifact_name
            if row is None:
                failures.append(f"manifest_missing_artifact:{artifact_name}")
                continue
            if not artifact_path.exists():
                failures.append(f"missing_artifact:{artifact_name}")
                continue
            actual_hash = hash_file(artifact_path)
            actual_size = artifact_path.stat().st_size
            if row.get("sha256") != actual_hash:
                failures.append(f"checksum_mismatch:{artifact_name}")
            if row.get("size_bytes") != actual_size:
                failures.append(f"size_mismatch:{artifact_name}")
            verified_artifacts.append(artifact_name)

    if registry:
        _check_false_flags(
            registry.get("execution", {}),
            failures,
            "neurovisual_dataset_registry.json",
            ("bulk_dataset_download", "a100_jobs_launched"),
        )
    if summary:
        _check_false_flags(
            summary.get("execution", {}),
            failures,
            "neurovisual_registry_verification_summary.json",
            ("bulk_dataset_download", "a100_jobs_launched", "cluster_jobs_launched"),
        )
    if adapter_plan:
        _check_false_flags(
            adapter_plan.get("execution", {}),
            failures,
            "neurovisual_adapter_plan.json",
            ("bulk_dataset_download", "a100_jobs_launched", "cluster_jobs_launched", "adapters_implemented"),
        )
    if query_plan:
        _check_false_flags(
            query_plan.get("execution", {}),
            failures,
            "neurovisual_metadata_query_plan.json",
            ("bulk_dataset_download", "a100_jobs_launched", "cluster_jobs_launched", "metadata_queries_executed"),
        )
        for target in query_plan.get("query_targets", []):
            if target.get("confirmed_accession") is not None:
                failures.append(f"query_plan_confirmed_accession:{target.get('target_id', '<unknown>')}")
            if target.get("verification_status") != "planned_not_executed":
                failures.append(f"query_plan_executed:{target.get('target_id', '<unknown>')}")
    if local_manifest_schema:
        _check_false_flags(
            local_manifest_schema.get("execution", {}),
            failures,
            "neurovisual_local_manifest_schema.json",
            (
                "adapters_implemented",
                "bulk_dataset_download",
                "a100_jobs_launched",
                "cluster_jobs_launched",
                "raw_file_existence_checked",
            ),
        )
    if split_audit_plan:
        _check_false_flags(
            split_audit_plan.get("execution", {}),
            failures,
            "neurovisual_split_audit_plan.json",
            (
                "split_audit_executed",
                "baselines_run",
                "models_run",
                "adapters_implemented",
                "bulk_dataset_download",
                "a100_jobs_launched",
                "cluster_jobs_launched",
            ),
        )
    if synthetic_split_audit:
        if synthetic_split_audit.get("schema") != "kahlus.nv1.local_split_audit.v1":
            failures.append("bad_schema:neurovisual_synthetic_split_audit.json")
        if not synthetic_split_audit.get("passed"):
            failures.append("synthetic_split_audit_failed")
        _check_false_flags(
            synthetic_split_audit.get("execution", {}),
            failures,
            "neurovisual_synthetic_split_audit.json",
            (
                "baselines_run",
                "models_run",
                "adapters_implemented",
                "bulk_dataset_download",
                "a100_jobs_launched",
                "cluster_jobs_launched",
                "raw_file_existence_checked",
            ),
        )
    if claim_gate:
        if claim_gate.get("schema") != "kahlus.nv1.claim_gate.v1":
            failures.append("bad_schema:neurovisual_registry_claim_gate.json")
        if not claim_gate.get("passed"):
            failures.append("claim_gate_failed")
        if claim_gate.get("blocked_claims_found"):
            failures.append("blocked_claims_found")

    return {
        "schema": "kahlus.nv1.registry_bundle_audit.v1",
        "artifact_dir": str(artifact_dir),
        "passed": not failures,
        "failures": failures,
        "verified_artifacts": verified_artifacts,
        "execution": {
            "bulk_dataset_download": False,
            "a100_jobs_launched": False,
            "cluster_jobs_launched": False,
            "metadata_queries_executed": False,
            "metadata_only": True,
        },
    }


def _read_json(path: Path, failures: list[str]) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        failures.append(f"missing_artifact:{path.name}")
        return {}
    except json.JSONDecodeError:
        failures.append(f"invalid_json:{path.name}")
        return {}
    if not isinstance(payload, dict):
        failures.append(f"invalid_json_object:{path.name}")
        return {}
    return payload


def _check_false_flags(
    execution: Any,
    failures: list[str],
    artifact_name: str,
    flag_names: tuple[str, ...],
) -> None:
    if not isinstance(execution, dict):
        failures.append(f"missing_execution:{artifact_name}")
        return
    for flag_name in flag_names:
        if execution.get(flag_name) is not False:
            failures.append(f"execution_flag_not_false:{artifact_name}:{flag_name}")


if __name__ == "__main__":
    raise SystemExit(main())
