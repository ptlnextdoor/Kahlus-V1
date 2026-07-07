#!/usr/bin/env python3
"""Run the local NV-1 evidence gate for downstream adapter readiness.

This orchestrates fixture replay, handoff manifest build, and handoff audit. It
does not check raw files, download datasets, execute adapters, run
baselines/models, or launch A100/cluster jobs.
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
    fixture_replay_dir = out_dir / "fixture_replay"
    handoff_path = out_dir / "neurovisual_handoff_manifest.json"
    gate_path = out_dir / "neurovisual_local_evidence_gate.json"
    report_path = out_dir / "neurovisual_local_evidence_gate.md"

    fixture_result = _run_local_script(
        "scripts/run_neurovisual_fixture_replay.py",
        "--out-dir",
        str(fixture_replay_dir),
    )
    handoff_result: subprocess.CompletedProcess[str] | None = None
    audit_result: subprocess.CompletedProcess[str] | None = None
    if fixture_result.returncode == 0:
        handoff_result = _run_local_script(
            "scripts/build_neurovisual_handoff_manifest.py",
            "--registry-package-dir",
            str(fixture_replay_dir / "registry_package"),
            "--fixture-replay-evidence",
            str(fixture_replay_dir / "neurovisual_fixture_replay_evidence.json"),
            "--out",
            str(handoff_path),
        )
    if handoff_result is not None and handoff_result.returncode == 0:
        audit_result = _run_local_script(
            "scripts/audit_neurovisual_handoff_manifest.py",
            "--handoff",
            str(handoff_path),
        )

    fixture_payload = _read_json_if_exists(fixture_replay_dir / "neurovisual_fixture_replay_evidence.json")
    handoff_payload = _read_json_if_exists(handoff_path)
    audit_payload = _json_object_from_stdout(audit_result.stdout if audit_result is not None else "")
    gate_payload = _gate_payload(
        out_dir=out_dir,
        fixture_replay_dir=fixture_replay_dir,
        handoff_path=handoff_path,
        fixture_result=fixture_result,
        handoff_result=handoff_result,
        audit_result=audit_result,
        fixture_payload=fixture_payload,
        handoff_payload=handoff_payload,
        audit_payload=audit_payload,
    )
    write_json(gate_path, gate_payload)
    report_path.write_text(_format_gate_report(gate_payload), encoding="utf-8")

    print(f"branch=nv1 out_dir={out_dir.resolve()}")
    print(f"fixture_replay_dir={fixture_replay_dir}")
    print(f"handoff_manifest={handoff_path}")
    print(f"evidence_gate={gate_path}")
    print(f"report={report_path}")
    print(f"passed={gate_payload['passed']}")
    print("bulk_dataset_download=false")
    print("a100_jobs_launched=false")
    return 0 if gate_payload["passed"] else 1


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


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _json_object_from_stdout(stdout: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _gate_payload(
    *,
    out_dir: Path,
    fixture_replay_dir: Path,
    handoff_path: Path,
    fixture_result: subprocess.CompletedProcess[str],
    handoff_result: subprocess.CompletedProcess[str] | None,
    audit_result: subprocess.CompletedProcess[str] | None,
    fixture_payload: dict[str, Any],
    handoff_payload: dict[str, Any],
    audit_payload: dict[str, Any],
) -> dict[str, Any]:
    handoff_returncode = handoff_result.returncode if handoff_result is not None else None
    audit_returncode = audit_result.returncode if audit_result is not None else None
    fixture_passed = bool(fixture_payload.get("passed"))
    handoff_passed = bool(handoff_payload.get("passed"))
    audit_passed = bool(audit_payload.get("passed"))
    passed = (
        fixture_result.returncode == 0
        and handoff_returncode == 0
        and audit_returncode == 0
        and fixture_passed
        and handoff_passed
        and audit_passed
    )
    return {
        "schema": "kahlus.nv1.local_evidence_gate.v1",
        "scope": "local NV-1 evidence gate before downstream adapter planning",
        "passed": passed,
        "out_dir": str(out_dir),
        "fixture_replay_dir": str(fixture_replay_dir),
        "handoff_manifest_path": str(handoff_path),
        "fixture_replay_returncode": fixture_result.returncode,
        "handoff_manifest_returncode": handoff_returncode,
        "handoff_audit_returncode": audit_returncode,
        "fixture_replay_passed": fixture_passed,
        "handoff_manifest_passed": handoff_passed,
        "handoff_audit_passed": audit_passed,
        "handoff_audit_failures": audit_payload.get("failures", []),
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


def _format_gate_report(payload: dict[str, Any]) -> str:
    execution = payload["execution"]
    assert isinstance(execution, dict)
    lines = [
        "# NV-1 Local Evidence Gate",
        "",
        f"- passed: {str(payload['passed']).lower()}",
        f"- fixture_replay_returncode: {payload['fixture_replay_returncode']}",
        f"- handoff_manifest_returncode: {payload['handoff_manifest_returncode']}",
        f"- handoff_audit_returncode: {payload['handoff_audit_returncode']}",
        f"- fixture_replay_passed: {str(payload['fixture_replay_passed']).lower()}",
        f"- handoff_manifest_passed: {str(payload['handoff_manifest_passed']).lower()}",
        f"- handoff_audit_passed: {str(payload['handoff_audit_passed']).lower()}",
        "",
        "## Execution Boundary",
        "",
        f"- bulk_dataset_download: {str(execution['bulk_dataset_download']).lower()}",
        f"- a100_jobs_launched: {str(execution['a100_jobs_launched']).lower()}",
        f"- cluster_jobs_launched: {str(execution['cluster_jobs_launched']).lower()}",
        f"- metadata_queries_executed: {str(execution['metadata_queries_executed']).lower()}",
        f"- adapters_implemented: {str(execution['adapters_implemented']).lower()}",
        f"- baselines_run: {str(execution['baselines_run']).lower()}",
        f"- models_run: {str(execution['models_run']).lower()}",
        f"- raw_file_existence_checked: {str(execution['raw_file_existence_checked']).lower()}",
        "",
        "This local evidence gate checks internal metadata and handoff consistency only; it does not approve adapters, baselines, models, clinical claims, or A100 execution.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
