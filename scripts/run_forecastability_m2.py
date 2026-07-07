#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from neurotwin.forecastability import run_m2_gate
from neurotwin.forecastability.m2 import download_sleep_edf_subset


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Kahlus Forecastability Trial 0 M2 sleep gate.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sleep-edf-root", default=None)
    parser.add_argument("--download-sleep-edf-subset", action="store_true")
    parser.add_argument("--sleep-edf-pairs", type=int, default=4)
    args = parser.parse_args()
    sleep_edf_root = args.sleep_edf_root
    if args.download_sleep_edf_subset:
        sleep_edf_root = sleep_edf_root or "/tmp/kahlus_sleep_edf_subset"
        download_sleep_edf_subset(Path(sleep_edf_root), n_pairs=args.sleep_edf_pairs)
    gate = run_m2_gate(args.out_dir, seed=args.seed, sleep_edf_root=sleep_edf_root)
    print(f"M2 gate passed: {gate['gate_passed']}")
    print(f"synthetic sleep machinery passed: {gate['synthetic_sleep_machinery_passed']}")
    print(f"report: {args.out_dir}/M2_EVIDENCE_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
