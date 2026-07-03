#!/usr/bin/env python3
"""Run Kahlus v1 EEG future-window baselines.

No A100, no cluster jobs, no dataset downloads. HBN-EEG support is local-path only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.eeg_v1 import (  # noqa: E402
    EEG_V1_CLAIM_SCOPE,
    audit_eeg_v1_split,
    build_future_forecasting_task,
    load_hbn_eeg_local_dataset,
    make_synthetic_eeg_v1_dataset,
    run_eeg_v1_autocorrelation_diagnostics,
    run_eeg_v1_baselines,
    write_eeg_v1_artifacts,
)
from neurotwin.eeg_v1.hbn import HBN_MISSING_ROOT  # noqa: E402
from neurotwin.eeg_v1.reporting import DEFAULT_EEG_V1_MODELS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("synthetic_fixture", "hbn_eeg"), required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window-length", type=int, default=8)
    parser.add_argument("--forecast-horizon", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=5)
    parser.add_argument("--models", default=",".join(DEFAULT_EEG_V1_MODELS))
    args = parser.parse_args()

    try:
        dataset = (
            make_synthetic_eeg_v1_dataset(seed=args.seed)
            if args.dataset == "synthetic_fixture"
            else load_hbn_eeg_local_dataset(args.data_root, seed=args.seed)
        )
    except FileNotFoundError as exc:
        print(HBN_MISSING_ROOT if args.dataset == "hbn_eeg" and args.data_root is None else str(exc))
        return 2

    try:
        task = build_future_forecasting_task(
            dataset,
            window_length=args.window_length,
            forecast_horizon=args.forecast_horizon,
        )
    except ValueError as exc:
        print(f"EEG v1 task configuration invalid: {exc}")
        return 3
    models = tuple(model.strip() for model in args.models.split(",") if model.strip())
    split_audit = audit_eeg_v1_split(dataset, split_type="subject_held_out")
    result = run_eeg_v1_baselines(task, seed=args.seed, train_steps=args.train_steps, model_ids=models)
    autocorrelation_diagnostics = run_eeg_v1_autocorrelation_diagnostics(
        dataset,
        seed=args.seed,
        window_length=args.window_length,
        forecast_horizon=args.forecast_horizon,
        train_steps=args.train_steps,
        model_ids=("persistence", "linear_ridge", "tiny_ssm"),
    )
    paths = write_eeg_v1_artifacts(
        args.out_dir,
        task=task,
        baseline_result=result,
        split_audit=split_audit,
        models=models,
        train_steps=args.train_steps,
        seed=args.seed,
        autocorrelation_diagnostics=autocorrelation_diagnostics,
    )

    print(f"branch=v1 dataset={args.dataset} out_dir={Path(args.out_dir).resolve()}")
    print(f"claim_scope={EEG_V1_CLAIM_SCOPE}")
    print(f"data_source={result['source']}")
    print(f"benchmark_status={result['benchmark_status']}")
    print(f"models_completed={sorted(result['metrics_by_model'])}")
    print(f"best_baseline={result['best_baseline']} best_baseline_mse={result['best_baseline_mse']}")
    print(f"kahlus_beats_best_baseline={result['kahlus_beats_best_baseline']}")
    metrics_payload = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    print(f"model_win_claim_allowed={metrics_payload['model_win_claim_allowed']}")
    print(f"model_win_status={metrics_payload['model_win_status']}")
    print(f"model_win_claim_failure_reasons={metrics_payload['model_win_claim_failure_reasons']}")
    target_scale = json.loads(paths["target_scale_context"].read_text(encoding="utf-8"))
    best_baseline_scale = target_scale["models"].get(result["best_baseline"] or "", {})
    print(f"target_units={target_scale['target_units']}")
    print(f"target_std={target_scale['target_std']}")
    print(f"target_variance={target_scale['target_variance']}")
    print(f"best_baseline_rmse_relative_to_target_std={best_baseline_scale.get('rmse_relative_to_target_std')}")
    autocorrelation_summary = autocorrelation_diagnostics["summary"]
    print(f"autocorrelation_warning={autocorrelation_summary['autocorrelation_warning']}")
    print(f"autocorrelation_short_horizon_best_mse={autocorrelation_summary['short_horizon_best_mse']}")
    print(f"autocorrelation_shuffled_target_best_mse={autocorrelation_summary['shuffled_target_best_mse']}")
    print(f"autocorrelation_tiny_ssm_mse={autocorrelation_summary['tiny_ssm_mse']}")
    print(f"autocorrelation_shuffled_target_control_mse={autocorrelation_summary['shuffled_target_control_mse']}")
    print(f"autocorrelation_persistence_or_ridge_dominates={autocorrelation_summary['persistence_or_ridge_dominates']}")
    print(f"autocorrelation_shuffled_target_close_to_real_baselines={autocorrelation_summary['shuffled_target_close_to_real_baselines']}")
    print(f"autocorrelation_long_horizon_delta_vs_short={autocorrelation_summary['long_horizon_delta_vs_short']}")
    print(f"autocorrelation_non_overlap_delta_vs_short={autocorrelation_summary['non_overlap_delta_vs_short']}")
    print(f"autocorrelation_delta_prediction_delta_vs_short={autocorrelation_summary['delta_prediction_delta_vs_short']}")
    print(f"autocorrelation_verdict={autocorrelation_summary['verdict']}")
    print(f"split_audit_passed={split_audit['leakage_passed']}")
    print(f"evidence_gate={paths['evidence_gate']}")
    print(f"baseline_verification={paths['verification']}")
    print("a100_jobs_launched=false")
    print(f"baseline_checksum_manifest={paths['checksum_manifest']}")
    print(
        "checksum_audit_command="
        f"PYTHONPATH=src python3 scripts/audit_eeg_v1_baseline_checksums.py --artifact-dir {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
