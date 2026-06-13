#!/usr/bin/env python3
"""Smoke runner: shared baselines on the Kahlus v3 synthetic Transition Gym.

PROPOSED / SYNTHETIC ONLY. Builds a Transition Gym instance (known hidden operators,
non-commutative locked perturbation battery, held-out composition split), runs baselines on
a response-profile forecasting task, and writes the artifact contract. No A100. No claim.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.baseline_runner import (  # noqa: E402
    DEFAULT_MODELS,
    run_baselines,
    transition_gym_regression_task,
    write_run_artifacts,
)
from neurotwin.transition_gym import SyntheticWorldConfig, build_transition_gym  # noqa: E402
from neurotwin.repro import write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=40)
    parser.add_argument("--n-episodes", type=int, default=48)
    parser.add_argument("--no-benchmark", action="store_true", help="Skip the v3 operator-recovery benchmark")
    args = parser.parse_args()

    cfg = SyntheticWorldConfig(seed=args.seed, n_episodes=args.n_episodes)
    bundle = build_transition_gym(cfg)
    task = transition_gym_regression_task(cfg)
    result = run_baselines(task, models=list(DEFAULT_MODELS), train_steps=args.train_steps, seed=args.seed)
    paths = write_run_artifacts(args.out_dir, task, result, train_steps=args.train_steps)
    data_card_path = write_json(Path(args.out_dir) / "data_card.json", bundle.data_card)

    print(f"branch=v3 task={task.name} out_dir={Path(args.out_dir).resolve()}")
    print(f"non_commutative={bundle.data_card['non_commutative']} gap={bundle.metadata['mean_commutator_gap']:.4f}")
    print(f"models_completed={sorted(result.metrics_by_model)}")
    print(f"failure_reasons={result.failure_reasons}")
    print(f"scientific_claim_allowed={result.evidence_gate['scientific_claim_allowed']}")
    print(f"data_card={data_card_path}")

    if not args.no_benchmark:
        from neurotwin.transition_gym import run_v3_benchmark, write_v3_report

        # Use an adequate episode budget for operator recovery (the CLI default is small).
        bench = run_v3_benchmark(seed=args.seed)
        bench_paths = write_v3_report(args.out_dir, bench)
        print("--- v3 operator-recovery benchmark ---")
        for outcome in bench.outcomes:
            print(f"  {'PASS' if outcome.passed else 'FAIL'} {outcome.name}")
        leaders = sorted(bench.leaderboard.items(), key=lambda kv: kv[1]["mse"])
        print("  leaderboard(mse):", {m: round(v["mse"], 4) for m, v in leaders})
        print(f"falsification_passed={bench.passed} scientific_claim_allowed={bench.gate['scientific_claim_allowed']}")
        print(f"ktm_beats_baselines={bench.ktm_beats_baselines} (untrained KTM scaffold; reported honestly)")
        print(f"v3_failure_reasons={bench.failure_reasons}")
        print(f"v3_report={bench_paths['report']} v3_evidence_gate={bench_paths['evidence_gate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
