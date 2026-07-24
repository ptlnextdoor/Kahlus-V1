#!/usr/bin/env python3
"""Refresh copy_trap_seeds.json/.csv from the synthetic EEG v1 copy-trap."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from neurotwin.benchmarks.baseline_suite import _metrics
from neurotwin.eeg_v1.dataset import build_future_forecasting_task, make_synthetic_eeg_v1_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> None:
    rows = []
    for seed in range(30):
        dataset = make_synthetic_eeg_v1_dataset(seed=seed, n_subjects=8, n_time=48, n_channels=3)
        task = build_future_forecasting_task(dataset, window_length=8, forecast_horizon=1, stride=2)
        x = np.asarray(task.x_test, dtype=np.float32)
        y = np.asarray(task.y_test, dtype=np.float32)
        h = int(task.metadata["forecast_horizon"])
        copy = np.empty_like(y)
        copy[:, :-h] = x[:, h:]
        copy[:, -h:] = x[:, -h:]
        unmasked = _metrics(y, copy, None, source_modality="eeg", target_modality="eeg", seed=0)["mse"]
        masked = _metrics(y, copy, task.metric_mask, source_modality="eeg", target_modality="eeg", seed=0)["mse"]
        rows.append(
            {
                "seed": seed,
                "unmasked_mse": float(unmasked),
                "masked_mse": float(masked),
                "ratio": float(masked / unmasked),
            }
        )

    payload = {
        "n": len(rows),
        "rows": rows,
        "mean_unmasked": float(np.mean([r["unmasked_mse"] for r in rows])),
        "mean_masked": float(np.mean([r["masked_mse"] for r in rows])),
        "mean_ratio": float(np.mean([r["ratio"] for r in rows])),
    }
    (DATA / "copy_trap_seeds.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    tidy = []
    for r in rows:
        tidy.append({"seed": r["seed"], "method": "unmasked_overlap", "mse": r["unmasked_mse"], "ratio": r["ratio"]})
        tidy.append({"seed": r["seed"], "method": "masked_future", "mse": r["masked_mse"], "ratio": r["ratio"]})
    with (DATA / "copy_trap_seeds.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "method", "mse", "ratio"])
        writer.writeheader()
        writer.writerows(tidy)

    print(
        f"n={payload['n']} mean_unmasked={payload['mean_unmasked']:.4f} "
        f"mean_masked={payload['mean_masked']:.4f} ratio={payload['mean_ratio']:.1f}"
    )


if __name__ == "__main__":
    main()
