#!/usr/bin/env python3
"""Run Kahlus v1 EEG autocorrelation and shuffled-target diagnostics.

No A100, no cluster jobs, and no dataset downloads. HBN-EEG support is local-path only.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.eeg_v1 import (  # noqa: E402
    load_hbn_eeg_local_dataset,
    make_synthetic_eeg_v1_dataset,
    run_eeg_v1_autocorrelation_diagnostics,
)
from neurotwin.eeg_v1.hbn import HBN_MISSING_ROOT  # noqa: E402
from neurotwin.repro import write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("synthetic_fixture", "hbn_eeg"), required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window-length", type=int, default=8)
    parser.add_argument("--forecast-horizon", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=5)
    parser.add_argument("--models", default="persistence,linear_ridge,tiny_ssm")
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

    model_ids = tuple(model.strip() for model in args.models.split(",") if model.strip())
    diagnostics = run_eeg_v1_autocorrelation_diagnostics(
        dataset,
        seed=args.seed,
        window_length=args.window_length,
        forecast_horizon=args.forecast_horizon,
        train_steps=args.train_steps,
        model_ids=model_ids,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_json(out_dir / "autocorrelation_diagnostics.json", diagnostics)
    csv_path = _write_csv(out_dir / "autocorrelation_diagnostics.csv", diagnostics)
    report_path = out_dir / "autocorrelation_diagnostics.md"
    report_path.write_text(_format_report(diagnostics), encoding="utf-8")

    summary = diagnostics["summary"]
    print(f"branch=v1 dataset={args.dataset} out_dir={out_dir.resolve()}")
    print(f"autocorrelation_diagnostics={json_path}")
    print(f"autocorrelation_diagnostics_csv={csv_path}")
    print(f"report={report_path}")
    print(f"tiny_ssm_mse={summary['tiny_ssm_mse']}")
    print(f"shuffled_target_control_mse={summary['shuffled_target_control_mse']}")
    print(f"persistence_or_ridge_dominates={summary['persistence_or_ridge_dominates']}")
    print(f"shuffled_target_close_to_real_baselines={summary['shuffled_target_close_to_real_baselines']}")
    print("bulk_dataset_download=false")
    print("a100_jobs_launched=false")
    return 0


def _write_csv(path: Path, diagnostics: dict[str, Any]) -> Path:
    fieldnames = (
        "diagnostic_id",
        "status",
        "best_model",
        "best_mse",
        "persistence_mse",
        "linear_ridge_mse",
        "tiny_ssm_mse",
        "shuffle_boundary",
        "train_targets_shuffled",
        "validation_targets_shuffled",
        "test_targets_shuffled",
        "leakage_passed",
        "reason",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in diagnostics.get("diagnostics", []):
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return path


def _format_report(diagnostics: dict[str, Any]) -> str:
    summary = diagnostics["summary"]
    lines = [
        "# EEG v1 Autocorrelation Diagnostics",
        "",
        "- scope: local v1 EEG scientific hardening",
        "- claim_scope: eeg_future_forecasting_benchmark_ready",
        "- a100_jobs_launched: false",
        "- bulk_dataset_download: false",
        "",
        "## Summary",
        "",
        f"- tiny_ssm_mse: {summary.get('tiny_ssm_mse')}",
        f"- shuffled_target_control_mse: {summary.get('shuffled_target_control_mse')}",
        f"- persistence_or_ridge_dominates: {summary.get('persistence_or_ridge_dominates')}",
        f"- shuffled_target_close_to_real_baselines: {summary.get('shuffled_target_close_to_real_baselines')}",
        f"- verdict: {summary.get('verdict')}",
        "",
        "## Diagnostics",
        "",
        "| diagnostic | status | best_model | best_mse | tiny_ssm_mse | shuffle_boundary |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in diagnostics.get("diagnostics", []):
        lines.append(
            "| "
            f"{row.get('diagnostic_id')} | "
            f"{row.get('status')} | "
            f"{row.get('best_model', '')} | "
            f"{row.get('best_mse', '')} | "
            f"{row.get('tiny_ssm_mse', '')} | "
            f"{row.get('shuffle_boundary', '')} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
