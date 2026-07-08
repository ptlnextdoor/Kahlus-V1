#!/usr/bin/env python3
"""Render publication-style EEG v1 ridge-baseline diagnostic figures.

The figures are grounded in the existing EEG v1 future-window benchmark. The
committed default uses the synthetic fixture so the docs can build everywhere.
Do not commit outputs produced from local public/raw EEG paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.benchmarks.baseline_suite import EXECUTABLE_BASELINE_RUNNERS  # noqa: E402
from neurotwin.eeg_v1 import (  # noqa: E402
    build_future_forecasting_task,
    load_hbn_eeg_local_dataset,
    make_synthetic_eeg_v1_dataset,
    run_eeg_v1_autocorrelation_diagnostics,
    run_eeg_v1_baselines,
)

KBLUE = "#0B3D91"
KTEAL = "#0F766E"
KGOLD = "#B7791F"
KRED = "#B42318"
KGREEN = "#146C43"
KGRAY = "#F5F7FA"
KINK = "#1F2937"
KMID = "#6B7280"
KLINE = "#D1D5DB"
BLACK = "#111827"

STYLE_SOURCE = "/Users/aayu/Downloads/versions/kahlus_v3_cna_master_dossier_2026-06-13_hybrid_visual_rebuild/generate_hybrid_figures.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("synthetic_fixture", "hbn_eeg"), default="synthetic_fixture")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window-length", type=int, default=8)
    parser.add_argument("--forecast-horizon", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=5)
    parser.add_argument("--example-index", type=int, default=0)
    args = parser.parse_args()

    apply_publication_style()
    dataset = (
        make_synthetic_eeg_v1_dataset(seed=args.seed)
        if args.dataset == "synthetic_fixture"
        else load_hbn_eeg_local_dataset(args.data_root, seed=args.seed)
    )
    task = build_future_forecasting_task(
        dataset,
        window_length=args.window_length,
        forecast_horizon=args.forecast_horizon,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    predictions = _predictions(task, seed=args.seed, train_steps=args.train_steps)
    result = run_eeg_v1_baselines(
        task,
        seed=args.seed,
        train_steps=args.train_steps,
        model_ids=("persistence", "linear_ridge", "autoregressive_ridge", "tiny_ssm"),
    )
    autocorr = run_eeg_v1_autocorrelation_diagnostics(
        dataset,
        seed=args.seed,
        window_length=args.window_length,
        forecast_horizon=args.forecast_horizon,
        train_steps=args.train_steps,
        model_ids=("persistence", "linear_ridge", "tiny_ssm"),
    )

    idx = min(max(0, int(args.example_index)), int(task.x_test.shape[0]) - 1)
    summary = _summary(args, task, result, autocorr, predictions, idx)

    figure_stems = {
        "window": "fig01_eeg_window_overlap_diagnostic",
        "feature_map": "fig02_ridge_design_matrix_contract",
        "prediction_overlay": "fig03_prediction_overlay_and_residuals",
        "baseline_controls": "fig04_baseline_and_autocorrelation_controls",
    }
    render_window_figure(task, idx, out / figure_stems["window"])
    render_feature_contract_figure(task, summary, out / figure_stems["feature_map"])
    render_prediction_overlay_figure(task, predictions, summary, idx, out / figure_stems["prediction_overlay"])
    render_baseline_controls_figure(summary, out / figure_stems["baseline_controls"])

    paths: dict[str, str] = {}
    for key, stem in figure_stems.items():
        paths[f"{key}_png"] = f"{stem}.png"
        paths[f"{key}_pdf"] = f"{stem}.pdf"
    summary["figure_files"] = paths
    summary["figure_style_source"] = STYLE_SOURCE

    (out / "eeg_v1_ridge_visual_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "eeg_v1_ridge_visual_analysis.md").write_text(_analysis_md(summary), encoding="utf-8")
    print(out / "eeg_v1_ridge_visual_analysis.md")
    return 0


def apply_publication_style() -> None:
    """Use the polished Kahlus dossier figure style from the versions archive."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 9.3,
            "axes.titlesize": 10.5,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 180,
            "savefig.dpi": 260,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _predictions(task: Any, *, seed: int, train_steps: int) -> dict[str, np.ndarray]:
    wanted = {"persistence", "linear_ridge", "autoregressive_ridge", "tiny_ssm"}
    out: dict[str, np.ndarray] = {}
    for spec in EXECUTABLE_BASELINE_RUNNERS:
        if spec.model_id not in wanted or not spec.supports(task):
            continue
        out[spec.model_id] = np.asarray(spec.predict(task, seed, train_steps), dtype=np.float64)
    return out


def _summary(args: argparse.Namespace, task: Any, result: dict[str, Any], autocorr: dict[str, Any], predictions: dict[str, np.ndarray], idx: int) -> dict[str, Any]:
    x = np.asarray(task.x_test[idx], dtype=np.float64)
    y = np.asarray(task.y_test[idx], dtype=np.float64)
    h = int(task.metadata["forecast_horizon"])
    overlap_corr = float(np.corrcoef(x[h:].ravel(), y[:-h].ravel())[0, 1]) if h < x.shape[0] else None
    return {
        "dataset": str(task.metadata.get("dataset_id")),
        "source": str(task.metadata.get("source")),
        "benchmark_status": str(task.metadata.get("benchmark_status")),
        "seed": int(args.seed),
        "example_index": int(idx),
        "window_length": int(task.metadata["window_length"]),
        "forecast_horizon": h,
        "sampling_rate_hz": task.metadata.get("sampling_rate_hz"),
        "n_train_windows": int(task.x_train.shape[0]),
        "n_test_windows": int(task.x_test.shape[0]),
        "ridge_feature_shape": list(np.asarray(task.x_train).reshape(task.x_train.shape[0], -1).shape),
        "ridge_target_shape": list(np.asarray(task.y_train).reshape(task.y_train.shape[0], -1).shape),
        "same_record_overlap_corr": overlap_corr,
        "metrics_by_model": result["metrics_by_model"],
        "best_baseline": result["best_baseline"],
        "autocorrelation_summary": autocorr["summary"],
        "example_mse": {
            model: float(np.mean((pred[idx] - y) ** 2))
            for model, pred in predictions.items()
        },
    }


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.06, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight="bold", color=KINK, va="top")


def clean(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")


def card(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], text: str, *, fc: str = KGRAY, ec: str = KBLUE, color: str = KINK, fontsize: float = 8.4, weight: str = "normal") -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.035", fc=fc, ec=ec, lw=1.0)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=color, weight=weight, linespacing=1.15)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str = KBLUE, lw: float = 1.4, rad: float = 0.0) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, lw=lw, color=color, connectionstyle=f"arc3,rad={rad}"))


def render_window_figure(task: Any, idx: int, stem: Path) -> None:
    x = np.asarray(task.x_test[idx], dtype=np.float64)
    y = np.asarray(task.y_test[idx], dtype=np.float64)
    h = int(task.metadata["forecast_horizon"])
    fs = float(task.metadata.get("sampling_rate_hz") or 1.0)
    channels = list(range(min(5, x.shape[1])))
    tx = np.arange(x.shape[0]) / fs
    ty = (np.arange(y.shape[0]) + h) / fs

    fig = plt.figure(figsize=(7.45, 4.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0], width_ratios=[1.2, 0.95], hspace=0.34, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    panel_label(ax, "A")
    scale = np.nanstd(np.concatenate([x[:, channels], y[:, channels]], axis=0)) or 1.0
    for row, ch in enumerate(channels):
        offset = row * 2.4
        ax.plot(tx, x[:, ch] / scale + offset, color=KBLUE, lw=1.1, label="input" if row == 0 else None)
        ax.plot(ty, y[:, ch] / scale + offset, color=KRED, lw=1.1, label="target" if row == 0 else None)
        ax.text(-0.006, offset, f"ch{ch}", ha="right", va="center", fontsize=7.8, color=KINK)
    ax.axvspan(tx[0], tx[-1], color=KBLUE, alpha=0.07)
    ax.axvspan(ty[0], ty[-1], color=KRED, alpha=0.07)
    ax.axvline(ty[0], color=KINK, ls="--", lw=0.8)
    ax.set_title("Short-horizon EEG window geometry", color=KBLUE, weight="bold")
    ax.set_xlabel("seconds from input-window start")
    ax.set_yticks([])
    ax.legend(loc="upper right", frameon=False, ncol=2)

    ax2 = fig.add_subplot(gs[1, 0])
    panel_label(ax2, "B")
    vmax = np.nanpercentile(np.abs(np.r_[x[:, channels].ravel(), y[:, channels].ravel()]), 98) or 1.0
    im = ax2.imshow(x[:, channels].T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax2.set_title("Input matrix $X_t$", color=KINK)
    ax2.set_xlabel("time sample")
    ax2.set_ylabel("channel")
    ax2.set_yticks(range(len(channels)), [f"ch{c}" for c in channels])
    fig.colorbar(im, ax=ax2, fraction=0.046, label="amplitude")

    ax3 = fig.add_subplot(gs[1, 1])
    clean(ax3)
    panel_label(ax3, "C")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    card(ax3, (0.05, 0.64), (0.90, 0.22), f"forecast horizon = {h} sample", fc="#FEF3C7", ec=KGOLD, weight="bold")
    card(ax3, (0.05, 0.35), (0.90, 0.22), f"overlap corr = {np.corrcoef(x[h:].ravel(), y[:-h].ravel())[0, 1]:.3f}", fc="#FEE2E2", ec=KRED, weight="bold")
    card(ax3, (0.05, 0.06), (0.90, 0.22), "interpret as diagnostic\nnot brain-state proof", fc="#DBEAFE", ec=KBLUE)
    ax3.text(0.0, -0.18, "Diagnostic refit. Synthetic fixture. Style derived from archived Kahlus dossier figures.", transform=ax3.transAxes, fontsize=7.2, color=KMID)
    save_figure(fig, stem)


def render_feature_contract_figure(task: Any, summary: dict[str, Any], stem: Path) -> None:
    w = int(task.x_train.shape[1])
    c = int(task.x_train.shape[2])
    fig, ax = plt.subplots(figsize=(7.55, 3.75))
    clean(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.94, "What linear_ridge receives", color=KBLUE, weight="bold", fontsize=11)
    card(ax, (0.05, 0.56), (0.22, 0.22), f"EEG window\n{w} time × {c} channels", fc="#DBEAFE", ec=KBLUE, weight="bold")
    card(ax, (0.39, 0.56), (0.22, 0.22), f"flatten\n{w*c} scalar features", fc=KGRAY, ec=KINK)
    card(ax, (0.73, 0.56), (0.22, 0.22), f"future target\n{w*c} scalar outputs", fc="#FEE2E2", ec=KRED, weight="bold")
    arrow(ax, (0.28, 0.67), (0.38, 0.67))
    arrow(ax, (0.62, 0.67), (0.72, 0.67), color=KRED)
    ax.add_patch(Rectangle((0.07, 0.17), 0.18, 0.23, fc="#DBEAFE", ec=KBLUE, lw=1.0))
    for i in range(min(w, 8)):
        ax.plot([0.07 + i * 0.18 / max(w, 1), 0.07 + i * 0.18 / max(w, 1)], [0.17, 0.40], color="white", lw=0.8)
    for j in range(min(c, 6)):
        ax.plot([0.07, 0.25], [0.17 + j * 0.23 / max(c, 1), 0.17 + j * 0.23 / max(c, 1)], color="white", lw=0.8)
    ax.text(0.16, 0.125, "$X_t$", ha="center", color=KBLUE, weight="bold")
    for i in range(min(w * c, 24)):
        x0 = 0.395 + (i % 12) * 0.016
        y0 = 0.24 + (i // 12) * 0.035
        ax.add_patch(Rectangle((x0, y0), 0.010, 0.022, fc="#DBEAFE", ec=KBLUE, lw=0.25))
    ax.text(0.50, 0.125, f"feature shape {summary['ridge_feature_shape']}", ha="center", color=KINK, fontsize=8)
    ax.add_patch(Rectangle((0.75, 0.17), 0.18, 0.23, fc="#FEE2E2", ec=KRED, lw=1.0))
    ax.text(0.84, 0.125, f"target shape {summary['ridge_target_shape']}", ha="center", color=KRED, fontsize=8)
    ax.text(0.02, 0.02, "Class B diagnostic refit. This is a matrix contract figure, not benchmark evidence or a clinical claim.", fontsize=7.5, color=KMID)
    save_figure(fig, stem)


def render_prediction_overlay_figure(task: Any, predictions: dict[str, np.ndarray], summary: dict[str, Any], idx: int, stem: Path) -> None:
    y = np.asarray(task.y_test[idx], dtype=np.float64)
    fs = float(task.metadata.get("sampling_rate_hz") or 1.0)
    h = int(task.metadata["forecast_horizon"])
    t = (np.arange(y.shape[0]) + h) / fs
    fig = plt.figure(figsize=(7.5, 4.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0], wspace=0.28, hspace=0.34)
    ax = fig.add_subplot(gs[0, :])
    panel_label(ax, "A")
    ch = 0
    ax.plot(t, y[:, ch], color=BLACK, lw=1.45, label="target")
    colors = {"persistence": KBLUE, "linear_ridge": KGREEN, "autoregressive_ridge": KGOLD, "tiny_ssm": KRED}
    for model in ("persistence", "linear_ridge", "autoregressive_ridge", "tiny_ssm"):
        if model in predictions:
            ax.plot(t, predictions[model][idx, :, ch], color=colors[model], lw=1.05, alpha=0.9, label=model)
    ax.set_title("Held-out synthetic fixture prediction overlay", color=KBLUE, weight="bold")
    ax.set_xlabel("seconds from input-window start")
    ax.set_ylabel("EEG amplitude (benchmark units)")
    ax.legend(frameon=False, ncol=3, loc="upper right")

    ax2 = fig.add_subplot(gs[1, 0])
    panel_label(ax2, "B")
    for model in ("persistence", "linear_ridge", "autoregressive_ridge", "tiny_ssm"):
        if model in predictions:
            residual = y[:, ch] - predictions[model][idx, :, ch]
            ax2.plot(t, residual, color=colors[model], lw=0.95, alpha=0.8, label=model)
    ax2.axhline(0, color=KLINE, lw=0.8)
    ax2.set_title("Residuals, channel 0")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("target - prediction")

    ax3 = fig.add_subplot(gs[1, 1])
    panel_label(ax3, "C")
    items = sorted(summary["example_mse"].items(), key=lambda x: x[1])
    labels = [k.replace("_", "\n") for k, _ in items]
    vals = [v for _, v in items]
    ax3.bar(np.arange(len(vals)), vals, color=[colors.get(k, KMID) for k, _ in items], alpha=0.88)
    ax3.set_xticks(np.arange(len(vals)), labels, rotation=0)
    ax3.set_ylabel("MSE")
    ax3.set_title("Example-window error")
    ax3.text(0.0, -0.35, "Lower is better. One example window only; use aggregate metrics for claims.", transform=ax3.transAxes, fontsize=7.2, color=KMID)
    save_figure(fig, stem)


def render_baseline_controls_figure(summary: dict[str, Any], stem: Path) -> None:
    fig = plt.figure(figsize=(7.55, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.34)
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "A")
    metrics = summary["metrics_by_model"]
    items = sorted(metrics.items(), key=lambda x: x[1]["mse"])
    labels = [k.replace("_", "\n") for k, _ in items]
    mse = [v["mse"] for _, v in items]
    r2 = [v["r2"] for _, v in items]
    x = np.arange(len(items))
    ax.bar(x - 0.18, mse, width=0.36, color=KBLUE, alpha=0.85, label="MSE")
    ax2 = ax.twinx()
    ax2.plot(x + 0.18, r2, color=KRED, marker="o", lw=1.4, label="$R^2$")
    ax.set_xticks(x, labels)
    ax.set_ylabel("MSE", color=KBLUE)
    ax2.set_ylabel("$R^2$", color=KRED)
    ax.set_title("Aggregate baseline comparison", color=KBLUE, weight="bold")
    lines, labs = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labs + labs2, frameon=False, loc="upper right")

    axc = fig.add_subplot(gs[0, 1])
    clean(axc)
    panel_label(axc, "B")
    axc.set_xlim(0, 1)
    axc.set_ylim(0, 1)
    ac = summary["autocorrelation_summary"]
    rows = [
        ("best short-horizon MSE", f"{ac.get('short_horizon_best_mse'):.3g}", KBLUE),
        ("long-horizon delta", f"+{ac.get('long_horizon_delta_vs_short'):.3g}", KRED),
        ("non-overlap delta", f"+{ac.get('non_overlap_delta_vs_short'):.3g}", KGOLD),
        ("shuffled control degrades", str(ac.get("shuffled_control_degrades")), KGREEN),
    ]
    y = 0.80
    for label, value, color in rows:
        card(axc, (0.05, y - 0.08), (0.90, 0.12), f"{label}\n{value}", fc="white", ec=color, color=KINK, fontsize=8.2, weight="bold" if color in (KRED, KBLUE) else "normal")
        y -= 0.18
    axc.text(0.05, 0.06, "Verdict: treat v1 as baseline infrastructure until harder controls are beaten.", fontsize=8.1, color=KRED, weight="bold", wrap=True)
    save_figure(fig, stem)


def _analysis_md(summary: dict[str, Any]) -> str:
    metrics = summary["metrics_by_model"]
    ac = summary["autocorrelation_summary"]
    figs = summary["figure_files"]
    lines = [
        "# EEG v1 Ridge Baseline Visual Sanity Check",
        "",
        "This is not a new benchmark. It visualizes the existing EEG v1 future-window benchmark so the ridge result is easier to inspect.",
        "",
        "```{admonition} Figure status",
        ":class: warning",
        "These are diagnostic refit figures generated from the synthetic fixture. They are useful for understanding benchmark geometry, not for public EEG or clinical claims.",
        "```",
        "",
        f"- dataset: `{summary['dataset']}`",
        f"- source: `{summary['source']}`",
        f"- benchmark_status: `{summary['benchmark_status']}`",
        f"- window_length: `{summary['window_length']}`",
        f"- forecast_horizon: `{summary['forecast_horizon']}`",
        f"- ridge feature matrix: `{summary['ridge_feature_shape']}` after flattening input windows",
        f"- ridge target matrix: `{summary['ridge_target_shape']}` after flattening future windows",
        f"- same-record shifted overlap correlation for the plotted test window: `{summary['same_record_overlap_corr']}`",
        f"- figure style source: `{summary['figure_style_source']}`",
        "",
        "## Publication-style diagnostic figures",
        "",
        f"![Window overlap diagnostic]({figs['window_png']})",
        "",
        f"[PDF version]({figs['window_pdf']})",
        "",
        f"![Ridge design matrix contract]({figs['feature_map_png']})",
        "",
        f"[PDF version]({figs['feature_map_pdf']})",
        "",
        f"![Prediction overlay and residuals]({figs['prediction_overlay_png']})",
        "",
        f"[PDF version]({figs['prediction_overlay_pdf']})",
        "",
        f"![Baseline and autocorrelation controls]({figs['baseline_controls_png']})",
        "",
        f"[PDF version]({figs['baseline_controls_pdf']})",
        "",
        "## Current benchmark metrics",
        "",
        "| model | mse | mae | r2 | pearsonr |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_id, row in sorted(metrics.items(), key=lambda item: item[1]["mse"]):
        lines.append(f"| {model_id} | {row['mse']:.6g} | {row['mae']:.6g} | {row['r2']:.6g} | {row['pearsonr']:.6g} |")
    lines += [
        "",
        "## Why ridge can look strong here",
        "",
        "- `linear_ridge` sees the raw EEG input window flattened into scalar time-channel features.",
        "- The target is the future EEG window, not a distant label. With `forecast_horizon=1`, most of the target window is the input window shifted by one sample.",
        "- The synthetic fixture is deliberately smooth/autoregressive, so a linear model is well matched to the data-generating structure.",
        "- Strong ridge or persistence performance is therefore a sanity-check signal for autocorrelation, short horizon, and overlap; it is not evidence of brain-state understanding.",
        "",
        "## Existing autocorrelation controls",
        "",
        f"- verdict: `{ac.get('verdict')}`",
        f"- persistence_or_ridge_dominates: `{ac.get('persistence_or_ridge_dominates')}`",
        f"- short_horizon_best_mse: `{ac.get('short_horizon_best_mse')}`",
        f"- long_horizon_delta_vs_short: `{ac.get('long_horizon_delta_vs_short')}`",
        f"- non_overlap_delta_vs_short: `{ac.get('non_overlap_delta_vs_short')}`",
        f"- shuffled_target_close_to_real_baselines: `{ac.get('shuffled_target_close_to_real_baselines')}`",
        "",
        "Bottom line: this analysis supports caution. The existing result is plausibly explained by local temporal structure and benchmark geometry, so adding more benchmarks would be noisier than first understanding this one.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
