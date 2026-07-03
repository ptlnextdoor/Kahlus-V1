#!/usr/bin/env python3
from __future__ import annotations

import argparse

from neurotwin.forecastability import run_m0_gate


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Kahlus Forecastability Trial 0 M0 gate.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=2)
    parser.add_argument("--allow-dirty", action="store_true", help="Do not fail the M0 gate on a dirty worktree.")
    args = parser.parse_args()
    gate = run_m0_gate(
        args.out_dir,
        seed=args.seed,
        train_steps=args.train_steps,
        enforce_clean_worktree=not args.allow_dirty,
    )
    print(f"M0 gate passed: {gate['gate_passed']}")
    print(f"report: {args.out_dir}/M0_EVIDENCE_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
