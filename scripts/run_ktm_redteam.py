#!/usr/bin/env python3
"""Local red-team falsifier for the KTM ``synthetic_ktm_recovery`` candidate (SYNTHETIC ONLY).

Runs the multi-seed + generator-family battery on CPU / single GPU (never A100) and writes
``ktm_redteam_report.{json,md}``. ``synthetic_ktm_recovery`` stays blocked unless the candidate
survives symmetric best-val selection, >=5 seeds with a positive lower bound, and generator-family
generalization. No real data, no cluster job, no model-superiority claim.
"""

from __future__ import annotations

import argparse

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.training_v3 import KTMTrainConfig  # noqa: E402
from neurotwin.training_v3.redteam_runner import (  # noqa: E402
    DEFAULT_FAMILIES,
    DEFAULT_SEEDS,
    run_redteam,
    write_redteam_report,
)


def _build_config(args: argparse.Namespace) -> KTMTrainConfig:
    from neurotwin.config import load_config

    payload: dict = dict(load_config(args.config)) if args.config else {}
    payload["mode"] = args.mode
    if args.steps is not None:
        payload["steps"] = args.steps
    return KTMTrainConfig.from_mapping(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", default="cpu_smoke", choices=["cpu_smoke", "single_gpu"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                        help="comma-separated seed list (default 0,1,2,3,4)")
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES),
                        help="comma-separated generator families (default linear,nonlinear,quadratic)")
    parser.add_argument("--margin", type=float, default=None)
    args = parser.parse_args()

    cfg = _build_config(args)
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip() != "")
    families = tuple(f.strip() for f in args.families.split(",") if f.strip() != "")

    report = run_redteam(cfg, seeds=seeds, families=families, margin=args.margin)
    paths = write_redteam_report(args.out_dir, report)

    g = report["redteam_gate"]
    summ = g["seed_summary"]
    aff = report["architecture_affinity"]
    print(f"recovery_allowed={report['recovery_allowed']}")
    print(f"blocker_reasons={report['blocker_reasons']}")
    print(f"seeds n={summ['n_seeds']} mean={summ['mean']:.4g} std={summ['std']:.4g} "
          f"lower_bound_95={summ['lower_bound_95']:.4g} n_positive={summ['n_positive']}")
    print(f"appears_generator_aligned={aff['appears_generator_aligned']} "
          f"by_family={aff['by_family_relative_improvement']}")
    print("report=" + ", ".join(f"{k}={v}" for k, v in paths.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
