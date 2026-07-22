#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from neurotwin.forecastability.passive_pci import run_passive_pci_gate


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Passive PCI state-discrimination gate.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/passive_pci_state"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sleep-edf-root",
        type=Path,
        default=None,
        help="Sleep-EDF cassette root for powered real cohort (omit for synthetic-only).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap on Sleep-EDF PSG/hypnogram pairs (default: full cohort).",
    )
    parser.add_argument("--bootstrap-mode", choices=["smoke", "claim"], default="smoke")
    args = parser.parse_args()
    gate = run_passive_pci_gate(
        args.out_dir,
        seed=args.seed,
        sleep_edf_root=args.sleep_edf_root,
        max_pairs=args.max_pairs,
        bootstrap_mode=args.bootstrap_mode,
    )
    print(f"passive pci gate passed: {gate['gate_passed']}")
    print(f"report: {args.out_dir}/PASSIVE_PCI_EVIDENCE_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
