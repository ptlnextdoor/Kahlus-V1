#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from neurotwin.forecastability.interoception_scout import run_interoception_rfs_gate


def main() -> int:
    parser = argparse.ArgumentParser(description="Run interoception RFS scout gate.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/interoception_rfs_scout"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sleep-edf-root",
        type=Path,
        default=None,
        help="Optional Sleep-EDF cassette root for real smoke (omit for synthetic-only).",
    )
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--bootstrap-mode", choices=["smoke", "claim"], default="smoke")
    args = parser.parse_args()
    gate = run_interoception_rfs_gate(
        args.out_dir,
        seed=args.seed,
        sleep_edf_root=args.sleep_edf_root,
        max_pairs=args.max_pairs,
        bootstrap_mode=args.bootstrap_mode,
    )
    print(f"interoception scout gate passed: {gate['gate_passed']}")
    print(f"report: {args.out_dir}/SCOUT_EVIDENCE_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
