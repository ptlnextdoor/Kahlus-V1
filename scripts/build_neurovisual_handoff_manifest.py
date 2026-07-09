#!/usr/bin/env python3
"""Build a claim-gated NV-1 handoff manifest from local evidence artifacts.

This summarizes the registry package and fixture replay evidence for downstream
adapter work. It does not check raw files, download datasets, execute adapters,
run baselines/models, or launch A100/cluster jobs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.neurovisual import evaluate_neurovisual_claim_gate  # noqa: E402
from neurotwin.repro import hash_file, write_json  # noqa: E402


REQUIRED_REGISTRY_PACKAGE_ARTIFACTS: tuple[str, ...] = (
    "neurovisual_dataset_registry.json",
    "neurovisual_registry_evidence_manifest.json",
    "neurovisual_registry_claim_gate.json",
    "neurovisual_synthetic_split_manifest.json",
    "neurovisual_synthetic_split_audit.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-package-dir", required=True)
    parser.add_argument("--fixture-replay-evidence", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    registry_package_dir = Path(args.registry_package_dir)
    fixture_replay_evidence_path = Path(args.fixture_replay_evidence)
    out_path = Path(args.out)
    payload = build_handoff_manifest(
        registry_package_dir=registry_package_dir,
        fixture_replay_evidence_path=fixture_replay_evidence_path,
        out_path=out_path,
    )
    write_json(out_path, payload)

    print(f"branch=nv1 handoff_manifest={out_path}")
    print(f"passed={payload['passed']}")
    print(f"claim_gate_passed={payload['claim_gate']['passed']}")
    print("bulk_dataset_download=false")
    print("a100_jobs_launched=false")
    return 0 if payload["passed"] else 1


def build_handoff_manifest(
    *,
    registry_package_dir: Path,
    fixture_replay_evidence_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    failures: list[str] = []
    registry = _read_json(registry_package_dir / "neurovisual_dataset_registry.json", failures)
    registry_evidence = _read_json(registry_package_dir / "neurovisual_registry_evidence_manifest.json", failures)
    registry_claim_gate = _read_json(registry_package_dir / "neurovisual_registry_claim_gate.json", failures)
    split_audit = _read_json(registry_package_dir / "neurovisual_synthetic_split_audit.json", failures)
    fixture_replay = _read_json(fixture_replay_evidence_path, failures)
    input_artifacts = _input_artifacts(registry_package_dir, fixture_replay_evidence_path, failures)
    registry_entries = registry.get("entries", []) if isinstance(registry, dict) else []
    confirmed_datasets = sum(1 for entry in registry_entries if entry.get("verification_status") == "confirmed")
    fixture_passed = bool(fixture_replay.get("passed")) if isinstance(fixture_replay, dict) else False
    split_audit_passed = bool(split_audit.get("passed")) if isinstance(split_audit, dict) else False
    registry_summary = {
        "schema": registry.get("schema") if isinstance(registry, dict) else None,
        "confirmed_datasets": confirmed_datasets,
        "total_registry_entries": len(registry_entries) if isinstance(registry_entries, list) else 0,
        "registry_claim_gate_passed": bool(registry_claim_gate.get("passed")) if isinstance(registry_claim_gate, dict) else False,
    }
    fixture_replay_summary = {
        "schema": fixture_replay.get("schema") if isinstance(fixture_replay, dict) else None,
        "passed": fixture_passed,
        "split_counts": fixture_replay.get("split_counts", {}) if isinstance(fixture_replay, dict) else {},
        "split_failures": fixture_replay.get("split_failures", []) if isinstance(fixture_replay, dict) else [],
    }
    claim_gate = evaluate_neurovisual_claim_gate(
        claim_scope="dataset_registry_ready",
        payloads=[registry_summary, fixture_replay_summary],
    )
    passed = not failures and claim_gate["passed"] and fixture_passed and split_audit_passed
    return {
        "schema": "kahlus.nv1.handoff_manifest.v1",
        "scope": "local NV-1 handoff manifest for downstream adapter planning",
        "passed": passed,
        "failures": failures + list(claim_gate["failure_reasons"]),
        "manifest_path": str(out_path),
        "registry_package_dir": str(registry_package_dir),
        "fixture_replay_evidence_path": str(fixture_replay_evidence_path),
        "claim_gate": claim_gate,
        "registry_summary": registry_summary,
        "fixture_replay_summary": fixture_replay_summary,
        "input_artifacts": input_artifacts,
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
        failures.append(f"missing_artifact:{path}")
        return {}
    except json.JSONDecodeError:
        failures.append(f"invalid_json:{path}")
        return {}
    if not isinstance(payload, dict):
        failures.append(f"invalid_json_object:{path}")
        return {}
    return payload


def _input_artifacts(
    registry_package_dir: Path,
    fixture_replay_evidence_path: Path,
    failures: list[str],
) -> list[dict[str, Any]]:
    paths = [registry_package_dir / artifact_name for artifact_name in REQUIRED_REGISTRY_PACKAGE_ARTIFACTS]
    paths.append(fixture_replay_evidence_path)
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            failures.append(f"missing_input_artifact:{path}")
            continue
        rows.append(
            {
                "path": path.name,
                "absolute_path": str(path),
                "sha256": hash_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows
if __name__ == "__main__":
    raise SystemExit(main())
