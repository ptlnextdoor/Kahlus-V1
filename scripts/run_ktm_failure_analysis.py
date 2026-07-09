#!/usr/bin/env python3
"""KTM-vs-SSM failure analysis for the Kahlus v3 KTM (PROPOSED / SYNTHETIC ONLY).

Diagnoses *why* the trained ``TorchKTM`` loses to the strongest matched baseline (``ssm_fallback``)
on the synthetic Transition Gym — per-horizon / per-perturbation / per-channel error, a KTM-vs-SSM
autopsy on the same held-out episodes, the loss-component breakdown, an optional small ablation
sweep, and a data-driven failure hypothesis. CPU / single-GPU friendly; no A100 is launched, no
cluster job is run, no scaling. The ``synthetic_ktm_recovery`` scope stays blocked — this tool only
reads a trained model and never relaxes a gate. No real data; no clinical / consciousness claim.

The full ablation sweep is opt-in (``--ablations``); the default command runs the fast base autopsy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.training_v3 import KTMTrainConfig  # noqa: E402
from neurotwin.training_v3.failure_analysis import (  # noqa: E402
    run_multiseed_objective_check,
    run_failure_analysis,
    write_failure_analysis,
    write_multiseed_objective_check,
)


def _build_config(args: argparse.Namespace) -> KTMTrainConfig:
    from neurotwin.config import load_config

    payload: dict = dict(load_config(args.config)) if args.config else {}
    payload["mode"] = args.mode
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.steps is not None:
        payload["steps"] = args.steps
    return KTMTrainConfig.from_mapping(payload)


def _parse_seeds(raw: str | None) -> list[int]:
    if raw is None or not raw.strip():
        return []
    seeds: list[int] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        seeds.append(int(text))
    if not seeds:
        raise ValueError("--seeds must include at least one integer seed")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--mode",
        default="cpu_smoke",
        choices=["cpu_smoke", "single_gpu"],
        help="local diagnostic mode only; distributed/cluster execution is intentionally blocked",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--ablations", action="store_true",
        help="opt-in: also run the small loss/architecture ablation sweep (slower)",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="optional comma-separated local seeds for the Sprint 3C full_objective vs traj_profile check",
    )
    args = parser.parse_args()

    cfg = _build_config(args)
    seeds = _parse_seeds(args.seeds)
    report = run_failure_analysis(cfg, run_ablations_flag=args.ablations)
    paths = write_failure_analysis(args.out_dir, report)
    if seeds:
        multiseed_report = run_multiseed_objective_check(cfg, seeds=seeds)
        paths.update(write_multiseed_objective_check(args.out_dir, multiseed_report))
    else:
        multiseed_report = None

    a = report["autopsy"]
    ov = a["overall"]
    print(f"branch=v3 mode={cfg.mode} out_dir={Path(args.out_dir).resolve()}")
    print(f"best_baseline={report['best_baseline']} ssm_beats_ktm={ov['ssm_beats_ktm']} "
          f"ktm_over_ssm_mse_ratio={ov['ratio_ktm_over_ssm']:.4g}")
    print(f"ktm_mse={ov['ktm']['mse']:.6g} ssm_mse={ov['ssm']['mse']:.6g} "
          f"rel_improvement={report['comparison']['relative_improvement']}")
    wins = a["where_ssm_beats_ktm"]
    print(f"where_ssm_beats_ktm perturbations={wins['perturbations']} horizons={wins['horizons']} "
          f"channels={wins['channels']}")
    print(f"where_ktm_beats_ssm perturbations={a['where_ktm_beats_ssm']['perturbations']}")
    lc = report["loss_components"]
    print(f"loss_components trajectory={lc['trajectory']:.4g} profile={lc['profile']:.4g} "
          f"nll={lc['nll']:.4g} total={lc['total']:.4g}")
    print(f"ablations_ran={report['ablations_ran']} n_ablations={len(report['ablations'])}")
    gap = report.get("objective_gap_comparison")
    if gap:
        print(
            "objective_gap_comparison "
            f"available={gap['available']} gap_narrowed={gap['gap_narrowed']} "
            f"reference={gap.get('reference_label')}:{gap.get('reference_ratio_ktm_over_ssm')} "
            f"best_candidate={gap.get('candidate_label')}:{gap.get('candidate_ratio_ktm_over_ssm')} "
            f"candidate_beats_ssm={gap.get('candidate_beats_ssm')} "
            f"candidate_beats_best_baseline_locked={gap.get('candidate_beats_best_baseline_locked')}"
        )
    if multiseed_report is not None:
        red = multiseed_report["relative_ratio_reduction_summary"]
        print(
            "multiseed_check "
            f"candidate={multiseed_report['candidate_label']} "
            f"gap_narrowed={multiseed_report['n_gap_narrowed']}/{multiseed_report['n_seeds']} "
            f"mean_relative_ratio_reduction={red['mean']} "
            f"any_candidate_beats_ssm={multiseed_report['any_candidate_beats_ssm']} "
            f"any_row_beats_best_baseline_locked={multiseed_report['any_row_beats_best_baseline_locked']} "
            f"recovery_claim_allowed={multiseed_report['recovery_claim_allowed']}"
        )
    print(f"best_failure_hypothesis: {report['best_failure_hypothesis']}")
    print(f"recovery_claim_allowed={report['recovery_claim_allowed']} (recovery stays blocked)")
    print("artifacts=" + ", ".join(f"{key}={value}" for key, value in paths.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
