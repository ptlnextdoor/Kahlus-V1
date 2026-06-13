#!/usr/bin/env python3
"""Smoke runner for the Kahlus v2 synthetic dual-field scaffold.

PROPOSED / SYNTHETIC ONLY. Simulates the dual-field system, runs the shared baselines on a
leakage-safe next-step EEG forecasting task, and writes the evidence artifact contract. The
evidence gate blocks any scientific claim (no calibration computed). No A100.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.baseline_runner import (  # noqa: E402
    DEFAULT_MODELS,
    dual_field_regression_task,
    run_baselines,
    write_run_artifacts,
)
from neurotwin.config import load_config  # noqa: E402
from neurotwin.models.dual_field import DualFieldConfig  # noqa: E402


def _config_from_yaml(path: str) -> tuple[DualFieldConfig, list[str], int, int, int]:
    payload = load_config(path)
    seed = int(payload.get("seed", 0))
    df_block = dict(payload.get("dual_field", {}))
    df_block.setdefault("seed", seed)
    config = DualFieldConfig(**df_block)
    baselines = dict(payload.get("baselines", {}))
    models = list(baselines.get("models", DEFAULT_MODELS))
    train_steps = int(baselines.get("train_steps", 60))
    window = int(payload.get("task", {}).get("window", 4))
    return config, models, train_steps, seed, window


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default=None, help="Optional dual_field YAML config")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=60)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--no-benchmark", action="store_true", help="Skip the v2 falsification benchmark")
    args = parser.parse_args()

    if args.config:
        config, models, train_steps, seed, window = _config_from_yaml(args.config)
    else:
        seed, train_steps, window = args.seed, args.train_steps, args.window
        config = DualFieldConfig(seed=seed)
        models = list(DEFAULT_MODELS)

    task = dual_field_regression_task(config, window=window)
    result = run_baselines(task, models=models, train_steps=train_steps, seed=seed)
    paths = write_run_artifacts(args.out_dir, task, result, models=models, train_steps=train_steps)

    print(f"branch=v2 task={task.name} out_dir={Path(args.out_dir).resolve()}")
    print(f"models_completed={sorted(result.metrics_by_model)}")
    print(f"failure_reasons={result.failure_reasons}")
    print(f"scientific_claim_allowed={result.evidence_gate['scientific_claim_allowed']}")
    print(f"evidence_gate={paths['evidence_gate']}")

    if not args.no_benchmark:
        from neurotwin.models.dual_field.benchmark import run_v2_benchmark, write_v2_report

        # Use the YAML config when provided; otherwise let the benchmark use its adequate
        # default data budget (the CLI default config is the tiny baseline-sweep size).
        bench = run_v2_benchmark(config if args.config else None, seed=seed)
        bench_paths = write_v2_report(args.out_dir, bench)
        print("--- v2 falsification benchmark ---")
        for outcome in bench.outcomes:
            print(f"  {'PASS' if outcome.passed else 'FAIL'} {outcome.name}")
        print(f"falsification_passed={bench.passed}")
        print(f"v2_failure_reasons={bench.failure_reasons}")
        print(f"v2_report={bench_paths['report']} v2_evidence_gate={bench_paths['evidence_gate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
