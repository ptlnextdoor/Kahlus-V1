# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
# ---
"""Ridge sanity-check waveform diagrams for the existing benchmark contract.

These figures are generated from the in-repo synthetic EEG-like benchmark arrays
used by ``neurotwin.benchmarks.baseline_suite``. They are intentionally labeled
as benchmark-contract diagnostics, not raw EEG evidence from the saved versions
archive, because the saved evidence bundles do not include raw EEG windows or
prediction arrays.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _figure_style as style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neurotwin.benchmarks.baseline_suite import _fit_ridge, _future_task, _make_paired_windows, _predict_persistence  # noqa: E402
from neurotwin.scoring.metrics import mse, pearsonr  # noqa: E402

FIGURES = ROOT / "figures"
DATA = ROOT / "data"
FIGURES.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)


def main() -> None:
    style.apply_style()
    bundle = make_example_bundle(seed=0, sample_index=0, channel_index=0)
    plot_future_window_contract(bundle, FIGURES / "FigureS6_ridge_future_window_contract")
    plot_prediction_overlay(bundle, FIGURES / "FigureS7_ridge_prediction_overlay")
    write_summary(bundle)


def make_example_bundle(seed: int, sample_index: int, channel_index: int) -> dict[str, object]:
    data = _make_paired_windows(seed)
    task = _future_task(data)
    ridge_prediction = _fit_ridge(task.x_train, task.y_train, task.x_test)
    persistence_prediction = _predict_persistence(task)
    x_test = np.asarray(task.x_test, dtype=np.float32)
    y_test = np.asarray(task.y_test, dtype=np.float32)
    sample_index = int(np.clip(sample_index, 0, x_test.shape[0] - 1))
    channel_index = int(np.clip(channel_index, 0, x_test.shape[-1] - 1))
    return {
        "seed": seed,
        "sample_index": sample_index,
        "channel_index": channel_index,
        "data": data,
        "task": task,
        "x": x_test[sample_index],
        "y": y_test[sample_index],
        "ridge_prediction": ridge_prediction[sample_index],
        "persistence_prediction": persistence_prediction[sample_index],
        "ridge_mse": float(mse(y_test, ridge_prediction)),
        "ridge_pearsonr": float(pearsonr(y_test, ridge_prediction)),
        "persistence_mse": float(mse(y_test, persistence_prediction)),
        "persistence_pearsonr": float(pearsonr(y_test, persistence_prediction)),
        "x_shape": tuple(int(v) for v in task.x_test.shape),
        "y_shape": tuple(int(v) for v in task.y_test.shape),
        "raw_generator_shape": tuple(int(v) for v in data["eeg"].shape),
    }


def plot_future_window_contract(bundle: dict[str, object], stem: Path) -> None:
    x = np.asarray(bundle["x"])
    y = np.asarray(bundle["y"])
    full = np.concatenate([x[:1], y], axis=0)
    n_time, n_channels = full.shape
    show_channels = min(4, n_channels)
    offsets = np.arange(show_channels)[::-1] * 5.0
    time = np.arange(n_time)

    fig = plt.figure(figsize=(7.35, 4.4), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], width_ratios=[1, 1])
    ax = fig.add_subplot(gs[0, :])
    ax.set_title("A. Future-state task: previous EEG time rows become ridge features", loc="left", fontweight="bold")
    ax.axvspan(-0.25, x.shape[0] - 0.75, color=style.LIGHT_BLUE, alpha=0.88, label="X input rows")
    ax.axvspan(0.75, n_time - 0.75, color=style.LIGHT_ORANGE, alpha=0.82, label="Y next-state rows")
    for idx in range(show_channels):
        ax.plot(time, full[:, idx] + offsets[idx], color=style.INK, linewidth=1.0)
        ax.scatter(np.arange(x.shape[0]), x[:, idx] + offsets[idx], s=14, color=style.BLUE, zorder=3)
        ax.scatter(np.arange(1, n_time), y[:, idx] + offsets[idx], s=14, color=style.ORANGE, zorder=3)
        ax.text(-0.45, offsets[idx], f"ch {idx}", ha="right", va="center", fontsize=7.2, color=style.GRAY)
    ax.annotate(
        "ridge sees these rows\nX = EEG[t0:t6, channels]",
        xy=(2.2, offsets[0] + 1.3),
        xytext=(0.2, offsets[0] + 5.0),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": style.BLUE},
        color=style.BLUE,
        fontsize=7.2,
    )
    ax.annotate(
        "target is one step later\nY = EEG[t1:t7, channels]",
        xy=(5.5, offsets[-1] - 1.0),
        xytext=(3.2, offsets[-1] - 5.0),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": style.ORANGE},
        color=style.ORANGE,
        fontsize=7.2,
    )
    ax.set_xlim(-0.65, n_time - 0.35)
    ax.set_xticks(time, [f"t{i}" for i in time])
    ax.set_yticks([])
    ax.set_xlabel("time index inside one benchmark window")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.grid(axis="x", color=style.LIGHT)

    ax_x = fig.add_subplot(gs[1, 0])
    ax_y = fig.add_subplot(gs[1, 1])
    vmax = float(np.nanmax(np.abs(np.concatenate([x, y])))) or 1.0
    sns.heatmap(x, ax=ax_x, cmap="cividis", center=0, vmin=-vmax, vmax=vmax, cbar=False, linewidths=0.4, linecolor="white")
    sns.heatmap(y, ax=ax_y, cmap="cividis", center=0, vmin=-vmax, vmax=vmax, cbar=True, cbar_kws={"label": "amplitude, a.u."}, linewidths=0.4, linecolor="white")
    ax_x.set_title("B. Design matrix slice X", loc="left", fontweight="bold", color=style.BLUE)
    ax_y.set_title("C. Target matrix slice Y", loc="left", fontweight="bold", color=style.ORANGE)
    for matrix_ax, label in ((ax_x, "feature rows t0:t6"), (ax_y, "target rows t1:t7")):
        matrix_ax.set_xlabel("channel")
        matrix_ax.set_ylabel(label)
        matrix_ax.set_xticklabels([f"ch {i}" for i in range(x.shape[1])], rotation=0)
        matrix_ax.set_yticklabels([f"r{i}" for i in range(x.shape[0])], rotation=0)
    fig.suptitle("Ridge input/target geometry from the existing benchmark code path", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.text(0.01, -0.01, "Diagnostic note: generated from baseline_suite._make_paired_windows(seed=0). The versions evidence archive does not currently save raw EEG windows.", fontsize=7.2, color=style.GRAY)
    style.save(fig, stem)


def plot_prediction_overlay(bundle: dict[str, object], stem: Path) -> None:
    x = np.asarray(bundle["x"])
    y = np.asarray(bundle["y"])
    ridge = np.asarray(bundle["ridge_prediction"])
    persistence = np.asarray(bundle["persistence_prediction"])
    ch = int(bundle["channel_index"])
    target_time = np.arange(1, y.shape[0] + 1)
    input_time = np.arange(x.shape[0])

    fig, axes = plt.subplots(2, 1, figsize=(7.35, 4.2), layout="constrained", height_ratios=[1.45, 0.75], sharex=False)
    ax = axes[0]
    ax.set_title("A. What ridge predicts for one channel", loc="left", fontweight="bold")
    ax.plot(input_time, x[:, ch], color=style.BLUE, linewidth=1.2, marker="o", markersize=3.2, label="input feature: EEG[t0:t6]")
    ax.plot(target_time, y[:, ch], color=style.ORANGE, linewidth=1.5, marker="o", markersize=3.2, label="true target: EEG[t1:t7]")
    ax.plot(target_time, ridge[:, ch], color=style.TEAL, linewidth=1.4, linestyle="--", marker="s", markersize=3.0, label="linear ridge prediction")
    ax.plot(target_time, persistence[:, ch], color=style.GRAY, linewidth=1.0, linestyle=":", label="persistence baseline")
    ax.axvspan(-0.2, x.shape[0] - 0.8, color=style.LIGHT_BLUE, alpha=0.55)
    ax.axvspan(0.8, y.shape[0] + 0.2, color=style.LIGHT_ORANGE, alpha=0.42)
    ax.annotate("same-channel recent value is highly informative\nfor a one-step/smooth target", xy=(3.0, x[3, ch]), xytext=(0.2, np.max(y[:, ch]) + 1.8), arrowprops={"arrowstyle": "->", "lw": 0.8, "color": style.INK}, fontsize=7.2, color=style.INK)
    ax.set_ylabel("amplitude, a.u.")
    ax.set_xticks(np.arange(0, y.shape[0] + 1), [f"t{i}" for i in range(y.shape[0] + 1)])
    ax.legend(frameon=False, loc="lower left", ncol=2)
    ax.grid(axis="both", color=style.LIGHT)

    residual = y[:, ch] - ridge[:, ch]
    ax_resid = axes[1]
    ax_resid.set_title("B. Ridge residual on this example target channel", loc="left", fontweight="bold")
    ax_resid.axhline(0, color=style.INK, linewidth=0.8)
    ax_resid.bar(target_time, residual, color=np.where(residual >= 0, style.ORANGE, style.TEAL), alpha=0.78, width=0.62)
    ax_resid.set_xlabel("target time index")
    ax_resid.set_ylabel("true - pred")
    ax_resid.set_xticks(target_time, [f"t{i}" for i in target_time])
    ax_resid.grid(axis="y", color=style.LIGHT)

    metrics = (
        f"synthetic benchmark arrays, seed={bundle['seed']} | "
        f"ridge MSE={bundle['ridge_mse']:.3f}, r={bundle['ridge_pearsonr']:.3f} | "
        f"persistence MSE={bundle['persistence_mse']:.3f}, r={bundle['persistence_pearsonr']:.3f}"
    )
    fig.suptitle("Ridge sanity check: prediction follows one-step EEG window geometry", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.text(0.01, -0.01, metrics + ". Not raw EEG evidence from saved versions bundles.", fontsize=7.2, color=style.GRAY)
    style.save(fig, stem)


def write_summary(bundle: dict[str, object]) -> None:
    summary = {
        "source": "neurotwin.benchmarks.baseline_suite._make_paired_windows(seed=0)",
        "claim_scope": "benchmark-contract diagnostic, not raw EEG evidence from versions bundles",
        "raw_generator_shape": list(bundle["raw_generator_shape"]),
        "x_test_shape": list(bundle["x_shape"]),
        "y_test_shape": list(bundle["y_shape"]),
        "sample_index": int(bundle["sample_index"]),
        "channel_index": int(bundle["channel_index"]),
        "ridge_mse": float(bundle["ridge_mse"]),
        "ridge_pearsonr": float(bundle["ridge_pearsonr"]),
        "persistence_mse": float(bundle["persistence_mse"]),
        "persistence_pearsonr": float(bundle["persistence_pearsonr"]),
        "contract": {
            "future_forecasting": "X = EEG window[:-1], Y = EEG window[1:]",
            "linear_ridge": "fit(flatten_time(X_train), flatten_time(Y_train)); reshape predictions to [sample, time, channel]",
        },
    }
    (DATA / "ridge_waveform_sanity_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
