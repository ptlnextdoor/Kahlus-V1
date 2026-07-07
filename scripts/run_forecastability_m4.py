#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from neurotwin.forecastability.m4 import (
    DEFAULT_HORIZONS,
    build_m4_sleep_edf_preregistration,
    run_m4_gate,
    run_m4_sleep_edf_primary_execution,
    _validate_m4_sleep_edf_preregistration,
)


DEFAULT_PREREGISTRATION = Path("configs/forecastability/m4_sleep_edf_primary_preregistration.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Kahlus Forecastability Trial 0 M4 gate.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    parser.add_argument("--primary-horizon", type=int, default=DEFAULT_HORIZONS[0])
    parser.add_argument("--sleep-edf-root", default=None)
    parser.add_argument("--sleep-edf-max-pairs", type=int, default=None)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--write-preregistration-only", action="store_true")
    parser.add_argument("--execute-full-sleep-edf", action="store_true")
    args = parser.parse_args()

    preregistration = _load_or_build_preregistration(args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.write_preregistration_only:
        _write_json(out / "m4_sleep_edf_preregistration.json", preregistration)
        print(f"preregistration: {out / 'm4_sleep_edf_preregistration.json'}")
        return 0

    if args.execute_full_sleep_edf:
        if not args.sleep_edf_root:
            parser.error("--execute-full-sleep-edf requires --sleep-edf-root")
        payload = run_m4_sleep_edf_primary_execution(
            out,
            sleep_edf_root=args.sleep_edf_root,
            preregistration=preregistration,
            repo_root=_repo_root(),
        )
        print(f"M4 Sleep-EDF primary execution gate passed: {payload['gate_passed']}")
        print(f"execution: {out / 'm4_sleep_edf_primary_execution.json'}")
        return 0

    gate = run_m4_gate(
        out,
        seed=args.seed,
        sleep_edf_root=args.sleep_edf_root,
        horizons=tuple(args.horizons),
        primary_horizon=args.primary_horizon,
    )
    print(f"M4 gate passed: {gate['gate_passed']}")
    print(f"synthetic gate passed: {gate['synthetic_gate_passed']}")
    print(f"report: {out / 'M4_EVIDENCE_REPORT.md'}")
    return 0


def _load_or_build_preregistration(args: argparse.Namespace) -> dict[str, object]:
    if args.preregistration.exists():
        preregistration = json.loads(args.preregistration.read_text(encoding="utf-8"))
    else:
        preregistration = build_m4_sleep_edf_preregistration(
            seed=args.seed,
            horizons=tuple(args.horizons),
            primary_horizon=args.primary_horizon,
            sleep_edf_max_pairs=args.sleep_edf_max_pairs,
        )
    _validate_m4_sleep_edf_preregistration(preregistration)
    return preregistration


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
