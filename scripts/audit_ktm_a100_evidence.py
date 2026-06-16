#!/usr/bin/env python3
"""Audit a returned KTM A100 evidence folder/zip (READ-ONLY, SYNTHETIC).

Consumes the evidence bundle a friend ships back after the synthetic KTM A100 micro-sweep and writes
a machine + human verdict: complete, secret-safe, GPU-consistent, claim-safe, scientifically
interpretable. It NEVER runs A100/cluster jobs, never changes KTM training, and never manufactures
recovery/model-superiority claims.

usage:
  PYTHONPATH=src python3 scripts/audit_ktm_a100_evidence.py \
    --evidence <path-to-evidence-zip-or-folder> --out-dir <out> --expected-gpus 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.a100_audit import audit_evidence, render_report_md  # noqa: E402
from neurotwin.repro import write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evidence", required=True, help="path to returned evidence folder or .zip")
    parser.add_argument("--out-dir", required=True, help="directory for audit JSON + markdown report")
    parser.add_argument("--expected-gpus", type=int, default=8, help="expected visible GPU count (e.g. 8, 7, 6, 1)")
    parser.add_argument(
        "--allow-missing-logs",
        action="store_true",
        help="do not warn when logs/*.log are absent (default: warn)",
    )
    args = parser.parse_args()

    result = audit_evidence(
        args.evidence,
        expected_gpus=args.expected_gpus,
        allow_missing_logs=args.allow_missing_logs,
    )

    out = Path(args.out_dir)
    audit_path = write_json(out / "a100_evidence_audit.json", result.to_dict())
    report_path = out / "a100_evidence_report.md"
    report_path.write_text(render_report_md(result), encoding="utf-8")

    fails = sum(1 for f in result.findings if f.severity == "fail")
    warns = sum(1 for f in result.findings if f.severity == "warn")
    print(f"verdict={result.verdict} fails={fails} warns={warns} expected_gpus={args.expected_gpus}")
    print(f"commit={result.commit_hash} run_completed={result.run_completed}")
    print(f"next_action={result.next_action}")
    print(f"audit_json={audit_path} report_md={report_path}")
    return 0 if result.verdict != "fail" else 2


if __name__ == "__main__":
    raise SystemExit(main())
