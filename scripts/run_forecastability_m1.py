#!/usr/bin/env python3
from __future__ import annotations

import argparse

from neurotwin.forecastability import run_m1_gate


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Kahlus Forecastability Trial 0 M1 gate.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    gate = run_m1_gate(args.out_dir, seed=args.seed)
    print(f"M1 gate passed: {gate['gate_passed']}")
    print(f"report: {args.out_dir}/M1_EVIDENCE_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
