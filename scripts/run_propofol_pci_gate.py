#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from neurotwin.forecastability.propofol_pci import run_propofol_pci_gate


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Propofol PCI state-discrimination gate (ds005620).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/propofol_pci_state"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ds-root",
        type=Path,
        default=None,
        help="OpenNeuro ds005620 root (omit for synthetic-only).",
    )
    parser.add_argument("--bootstrap-mode", choices=["smoke", "claim"], default="smoke")
    args = parser.parse_args()
    gate = run_propofol_pci_gate(
        args.out_dir,
        seed=args.seed,
        ds_root=args.ds_root,
        bootstrap_mode=args.bootstrap_mode,
    )
    print(f"propofol pci gate passed: {gate['gate_passed']}")
    print(f"report: {args.out_dir}/PROPOFOL_PCI_EVIDENCE_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
