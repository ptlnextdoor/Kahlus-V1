#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from neurotwin.forecastability import run_m3_gate
from neurotwin.forecastability.m3 import download_chbmit_subset


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Kahlus Forecastability Trial 0 M3 seizure gate.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chbmit-root", default=None)
    parser.add_argument("--download-chbmit-subset", action="store_true")
    args = parser.parse_args()
    chbmit_root = args.chbmit_root
    if args.download_chbmit_subset:
        chbmit_root = chbmit_root or "/tmp/kahlus_chbmit_subset"
        download_chbmit_subset(Path(chbmit_root))
    gate = run_m3_gate(args.out_dir, seed=args.seed, chbmit_root=chbmit_root)
    print(f"M3 gate passed: {gate['gate_passed']}")
    print(f"forecastability class: {gate['forecastability_class']}")
    print(f"report: {args.out_dir}/M3_EVIDENCE_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
