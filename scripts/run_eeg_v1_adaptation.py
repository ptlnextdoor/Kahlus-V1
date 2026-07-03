#!/usr/bin/env python3
"""Run local Kahlus v1 few-shot EEG adaptation comparisons.

This is a CPU/local validation lane only. It does not launch A100, torchrun, or cluster jobs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.eeg_v1 import (  # noqa: E402
    EEG_V1_ADAPTATION_CLAIM_SCOPE,
    audit_eeg_v1_split,
    build_fewshot_adaptation_task,
    load_hbn_eeg_local_dataset,
    make_synthetic_eeg_v1_dataset,
    run_fewshot_adaptation,
    write_fewshot_adaptation_artifacts,
)
from neurotwin.eeg_v1.hbn import HBN_MISSING_ROOT  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("synthetic_fixture", "hbn_eeg"), required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window-length", type=int, default=8)
    parser.add_argument("--forecast-horizon", type=int, default=1)
    parser.add_argument("--support-windows", type=int, default=5)
    parser.add_argument("--pretrain-steps", type=int, default=20)
    parser.add_argument("--adapt-steps", type=int, default=10)
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

    split_audit = audit_eeg_v1_split(dataset, split_type="subject_held_out")
    try:
        task = build_fewshot_adaptation_task(
            dataset,
            window_length=args.window_length,
            forecast_horizon=args.forecast_horizon,
            support_windows=args.support_windows,
        )
    except ValueError as exc:
        print(f"EEG v1 adaptation task configuration invalid: {exc}")
        return 3
    result = run_fewshot_adaptation(
        task,
        seed=args.seed,
        pretrain_steps=args.pretrain_steps,
        adapt_steps=args.adapt_steps,
    )
    paths = write_fewshot_adaptation_artifacts(
        args.out_dir,
        task=task,
        result=result,
        split_audit=split_audit,
    )

    best = result["adaptation_ranking"][0] if result["adaptation_ranking"] else {"method": "none", "value": float("nan")}
    target_scale = json.loads(paths["target_scale_context"].read_text(encoding="utf-8"))
    baseline_gap = json.loads(paths["baseline_gap_summary"].read_text(encoding="utf-8"))
    subject_baseline_gap = json.loads(paths["subject_baseline_gap_summary"].read_text(encoding="utf-8"))
    best_scale = target_scale["methods"].get(best["method"], {})
    subject_metric_rows = len(result.get("subject_metrics", ()))
    print(f"branch=v1 dataset={args.dataset} out_dir={Path(args.out_dir).resolve()}")
    print(f"claim_scope={EEG_V1_ADAPTATION_CLAIM_SCOPE}")
    print(f"data_source={result['source']}")
    print(f"benchmark_status={result['benchmark_status']}")
    print(f"subjects={list(task.subjects)} support_windows={task.support_windows} query_windows={task.query_windows}")
    print(f"best_method={best['method']} best_mse={best['value']}")
    print(f"target_units={target_scale['target_units']}")
    print(f"target_std={target_scale['target_std']}")
    print(f"target_variance={target_scale['target_variance']}")
    print(f"best_method_rmse_relative_to_target_std={best_scale.get('rmse_relative_to_target_std')}")
    print(f"best_support_baseline={baseline_gap['best_support_baseline']}")
    print(f"best_adaptation_method={baseline_gap['best_adaptation_method']}")
    print(f"adaptation_vs_best_support_baseline_mse_delta={baseline_gap['adaptation_vs_best_support_baseline_mse_delta']}")
    print(f"adaptation_beats_best_support_baseline={baseline_gap['adaptation_beats_best_support_baseline']}")
    print(f"adaptation_subject_rows={subject_metric_rows}")
    print(f"adaptation_subject_metrics={paths['subject_metrics']}")
    print(
        "subjects_where_adaptation_beats_best_support_baseline="
        f"{subject_baseline_gap['subjects_where_adaptation_beats_best_support_baseline']}"
    )
    print(f"adaptation_subject_baseline_gap_summary={paths['subject_baseline_gap_summary']}")
    print(f"adaptation_verification={paths['verification']}")
    print(f"adaptation_checksum_manifest={paths['checksum_manifest']}")
    print(
        "checksum_audit_command="
        f"PYTHONPATH=src python3 scripts/audit_eeg_v1_adaptation_checksums.py --artifact-dir {args.out_dir}"
    )
    print("a100_jobs_launched=false")
    print(f"evidence_gate={paths['evidence_gate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
