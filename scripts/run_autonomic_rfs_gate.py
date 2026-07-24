#!/usr/bin/env python3
"""Run autonomic RFS gate (MESA primary, SHHS dataset-held-out secondary)."""
from __future__ import annotations

import argparse
from pathlib import Path

from neurotwin.forecastability.autonomic_rfs import run_autonomic_rfs_gate

DEFAULT_MESA = Path("/Users/aayu/datasets/kahlus_multidataset_public/nsrr/mesa")
DEFAULT_SHHS = Path("/Users/aayu/datasets/kahlus_multidataset_public/nsrr/shhs")
DEFAULT_OUT = Path("artifacts/autonomic_rfs_arousal")


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomic RFS arousal gate (NSRR MESA/SHHS).")
    parser.add_argument("--mesa-root", type=Path, default=DEFAULT_MESA)
    parser.add_argument("--shhs-root", type=Path, default=DEFAULT_SHHS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-mode", choices=("smoke", "claim"), default="smoke")
    parser.add_argument("--min-subjects", type=int, default=8)
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    mesa_root = args.mesa_root if args.mesa_root.is_dir() else None
    shhs_root = args.shhs_root if args.shhs_root.is_dir() else None
    gate = run_autonomic_rfs_gate(
        args.out_dir,
        seed=args.seed,
        mesa_root=mesa_root,
        shhs_root=shhs_root,
        bootstrap_mode=args.bootstrap_mode,
        max_recordings=args.max_recordings,
        min_subjects=args.min_subjects,
    )
    print(f"autonomic rfs gate passed: {gate['gate_passed']}")
    print(f"report: {args.out_dir / 'AUTONOMIC_RFS_EVIDENCE_REPORT.md'}")
    return 0 if gate["gate_passed"] or gate["mesa_status"] == "skipped" else 1


if __name__ == "__main__":
    raise SystemExit(main())
