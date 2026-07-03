#!/usr/bin/env python3
"""Local self-smoke for an extracted Kahlus 7xA100 runner.

This script never launches A100, Slurm, Docker, torchrun, or dataset downloads. It checks that
the packaged evidence writer and returned-evidence auditor can run together on a tiny incomplete
fixture and fail closed with report artifacts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    commit = _commit_hash(root)
    env = {**os.environ, "PYTHONPATH": "src"}
    with tempfile.TemporaryDirectory(prefix="kahlus_a100_runner_smoke_") as tmp:
        tmp_root = Path(tmp)
        persistent = tmp_root / "persistent"
        (persistent / "runs" / "moabb_a100_smoke").mkdir(parents=True)
        evidence_zip = tmp_root / "runner-self-smoke-evidence.zip"
        evidence_name = "runner-self-smoke-evidence"
        package_result = _run(
            [
                sys.executable,
                "scripts/package_a100_evidence_bundle.py",
                str(persistent),
                str(evidence_zip),
                evidence_name,
                str(root),
                commit,
            ],
            cwd=root,
            env=env,
        )
        if package_result.returncode != 0:
            return _fail("evidence_package_failed", package_result)
        if not evidence_zip.is_file():
            return _fail_text("evidence_zip_missing")
        with zipfile.ZipFile(evidence_zip, "r") as archive:
            names = set(archive.namelist())
        for rel in (
            f"{evidence_name}/README_HANDOFF.md",
            f"{evidence_name}/README_SEND_TO_FRIEND.md",
            f"{evidence_name}/handoff-SHA256SUMS",
        ):
            if rel not in names:
                return _fail_text(f"evidence_member_missing={rel}")

        audit_out = tmp_root / "audit"
        audit_result = _run(
            [
                sys.executable,
                "scripts/audit_ktm_a100_evidence.py",
                "--evidence",
                str(evidence_zip),
                "--out-dir",
                str(audit_out),
                "--expected-gpus",
                "7",
                "--allow-missing-logs",
            ],
            cwd=root,
            env=env,
        )
        if audit_result.returncode != 2:
            return _fail("audit_did_not_fail_closed", audit_result)
        audit_json = audit_out / "a100_evidence_audit.json"
        audit_md = audit_out / "a100_evidence_report.md"
        if not audit_json.is_file() or not audit_md.is_file():
            return _fail_text("audit_artifacts_missing")
        payload = json.loads(audit_json.read_text(encoding="utf-8"))
        codes = {finding.get("code") for finding in payload.get("findings", []) if isinstance(finding, dict)}
        if payload.get("verdict") != "fail" or "required_file_missing" not in codes:
            return _fail_text("audit_verdict_unexpected")

    print("runner_self_smoke_passed=true")
    print("a100_jobs_launched=false")
    print("torchrun_launched=false")
    print("cluster_jobs_launched=false")
    return 0


def _commit_hash(root: Path) -> str:
    path = root / "COMMIT_HASH.txt"
    if path.is_file():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    return "unknown"


def _run(args: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True)


def _fail(code: str, result: subprocess.CompletedProcess[str]) -> int:
    print(f"runner_self_smoke_passed=false code={code}", file=sys.stderr)
    if result.stdout:
        print(result.stdout, file=sys.stderr)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return 2


def _fail_text(message: str) -> int:
    print(f"runner_self_smoke_passed=false code={message}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
