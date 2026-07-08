#!/usr/bin/env python3
"""Render EEG v1 ridge-baseline sanity-check diagrams.

Uses the existing EEG v1 future-window benchmark. The default synthetic fixture is safe to commit;
do not commit outputs produced from local public/raw EEG paths.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np

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
    paths = {
        "waveform": out / "eeg_v1_ridge_waveform_window.svg",
        "feature_map": out / "eeg_v1_ridge_feature_map.svg",
        "prediction_overlay": out / "eeg_v1_ridge_prediction_overlay.svg",
        "analysis": out / "eeg_v1_ridge_visual_analysis.md",
        "summary": out / "eeg_v1_ridge_visual_summary.json",
    }
    paths["waveform"].write_text(_waveform_svg(task, idx), encoding="utf-8")
    paths["feature_map"].write_text(_feature_map_svg(task), encoding="utf-8")
    paths["prediction_overlay"].write_text(_prediction_overlay_svg(task, predictions, idx), encoding="utf-8")
    summary = _summary(args, task, result, autocorr, predictions, idx)
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["analysis"].write_text(_analysis_md(summary, paths), encoding="utf-8")
    print(paths["analysis"])
    return 0


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


def _waveform_svg(task: Any, idx: int) -> str:
    x = np.asarray(task.x_test[idx], dtype=np.float64)
    y = np.asarray(task.y_test[idx], dtype=np.float64)
    h = int(task.metadata["forecast_horizon"])
    fs = float(task.metadata.get("sampling_rate_hz") or 1.0)
    times_x = np.arange(x.shape[0]) / fs
    times_y = (np.arange(y.shape[0]) + h) / fs
    series = [(times_x, x[:, ch], f"input ch{ch}", "#1565c0") for ch in range(min(3, x.shape[1]))]
    series += [(times_y, y[:, ch], f"target ch{ch}", "#c2410c") for ch in range(min(3, y.shape[1]))]
    return _line_svg(
        "Raw EEG window used by ridge",
        series,
        annotations=[
            f"linear_ridge features = current window: {x.shape[0]} time samples x {x.shape[1]} channels, flattened",
            f"target = future window shifted by forecast_horizon={h}; most samples overlap at short horizon",
        ],
        x_label="seconds from input-window start",
        y_label="EEG amplitude (benchmark units)",
    )


def _prediction_overlay_svg(task: Any, predictions: dict[str, np.ndarray], idx: int) -> str:
    x = np.asarray(task.x_test[idx], dtype=np.float64)
    y = np.asarray(task.y_test[idx], dtype=np.float64)
    h = int(task.metadata["forecast_horizon"])
    fs = float(task.metadata.get("sampling_rate_hz") or 1.0)
    ch = 0
    times_x = np.arange(x.shape[0]) / fs
    times_y = (np.arange(y.shape[0]) + h) / fs
    series = [
        (times_x, x[:, ch], "input ch0", "#94a3b8"),
        (times_y, y[:, ch], "target ch0", "#111827"),
    ]
    colors = {"persistence": "#2563eb", "linear_ridge": "#16a34a", "autoregressive_ridge": "#9333ea", "tiny_ssm": "#dc2626"}
    for model in ("persistence", "linear_ridge", "autoregressive_ridge", "tiny_ssm"):
        if model in predictions:
            series.append((times_y, predictions[model][idx, :, ch], model, colors[model]))
    return _line_svg(
        "Example prediction overlay",
        series,
        annotations=[
            "Overlay is one held-out test window, channel 0.",
            "If ridge tracks the target here, it may be using smooth short-horizon autocorrelation, not deep structure.",
        ],
        x_label="seconds from input-window start",
        y_label="EEG amplitude (benchmark units)",
    )


def _feature_map_svg(task: Any) -> str:
    w = int(task.x_train.shape[1])
    c = int(task.x_train.shape[2])
    h = int(task.metadata["forecast_horizon"])
    rows: list[str] = []
    width, height = 900, 430
    rows.append(_svg_open(width, height))
    rows.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    rows.append(_text(30, 38, "What linear_ridge actually receives", 22, "#111827", "bold"))
    rows.append(_text(30, 70, f"Input X is a {w} x {c} raw EEG window; ridge flattens it to {w*c} scalar features.", 14))
    rows.append(_text(30, 94, f"Target Y is the future {w} x {c} EEG window, shifted by horizon={h}; ridge predicts all {w*c} target scalars.", 14))
    _grid(rows, 40, 135, w, c, "Input feature matrix X[t, channel]", "#dbeafe", "#1d4ed8")
    _arrow(rows, 355, 210, 495, 210, "flatten")
    rows.append(_text(515, 155, "x = [", 16, "#111827", "bold"))
    for i in range(w * c):
        x = 555 + (i % 12) * 22
        y = 145 + (i // 12) * 22
        rows.append(f'<rect x="{x}" y="{y}" width="16" height="16" rx="2" fill="#dbeafe" stroke="#60a5fa"/>')
    rows.append(_text(820, 155, "]", 16, "#111827", "bold"))
    _arrow(rows, 640, 240, 640, 292, "ridge coefficients")
    _grid(rows, 500, 320, w, c, "Predicted future Y[t+h, channel]", "#ffedd5", "#c2410c")
    rows.append(_text(40, 385, "Sanity-check interpretation: short horizon + overlapping windows makes much of Y a shifted copy of X.", 14, "#7c2d12"))
    rows.append("</svg>")
    return "\n".join(rows)


def _line_svg(title: str, series: list[tuple[np.ndarray, np.ndarray, str, str]], *, annotations: list[str], x_label: str, y_label: str) -> str:
    width, height = 900, 460
    ml, mr, mt, mb = 75, 25, 55, 92
    xs = np.concatenate([np.asarray(s[0], dtype=np.float64) for s in series])
    ys = np.concatenate([np.asarray(s[1], dtype=np.float64) for s in series])
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    pad = max(1e-6, (ymax - ymin) * 0.12)
    ymin, ymax = ymin - pad, ymax + pad

    def sx(v: float) -> float:
        return ml + (v - xmin) / max(1e-9, xmax - xmin) * (width - ml - mr)

    def sy(v: float) -> float:
        return mt + (ymax - v) / max(1e-9, ymax - ymin) * (height - mt - mb)

    rows = [_svg_open(width, height), '<rect width="100%" height="100%" fill="#ffffff"/>']
    rows.append(_text(30, 34, title, 22, "#111827", "bold"))
    rows.append(f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#111827"/>')
    rows.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#111827"/>')
    rows.append(_text(width / 2 - 80, height - 24, x_label, 13, "#374151"))
    rows.append(_text(12, 215, y_label, 13, "#374151"))
    for tx in np.linspace(xmin, xmax, 5):
        rows.append(f'<line x1="{sx(tx):.1f}" y1="{height-mb}" x2="{sx(tx):.1f}" y2="{height-mb+5}" stroke="#374151"/>')
        rows.append(_text(sx(tx) - 16, height - mb + 22, f"{tx:.3f}", 11, "#374151"))
    for ty in np.linspace(ymin, ymax, 5):
        rows.append(f'<line x1="{ml-5}" y1="{sy(ty):.1f}" x2="{ml}" y2="{sy(ty):.1f}" stroke="#374151"/>')
        rows.append(_text(8, sy(ty) + 4, f"{ty:.2f}", 11, "#374151"))
    legend_x, legend_y = 655, 54
    for i, (_, _, label, color) in enumerate(series):
        y = legend_y + i * 18
        rows.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+22}" y2="{y}" stroke="{color}" stroke-width="2.4"/>')
        rows.append(_text(legend_x + 28, y + 4, label, 12, "#111827"))
    for xs_i, ys_i, _, color in series:
        pts = " ".join(f"{sx(float(x)):.1f},{sy(float(y)):.1f}" for x, y in zip(xs_i, ys_i))
        rows.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.0" points="{pts}"/>')
    for i, note in enumerate(annotations):
        rows.append(_text(75, height - 58 + i * 18, note, 13, "#7c2d12"))
    rows.append("</svg>")
    return "\n".join(rows)


def _grid(rows: list[str], x0: int, y0: int, time_steps: int, channels: int, label: str, fill: str, stroke: str) -> None:
    rows.append(_text(x0, y0 - 15, label, 13, "#111827", "bold"))
    cell = 22
    for t in range(time_steps):
        for ch in range(channels):
            rows.append(f'<rect x="{x0 + t*cell}" y="{y0 + ch*cell}" width="20" height="20" rx="2" fill="{fill}" stroke="{stroke}"/>')
            if t == 0:
                rows.append(_text(x0 - 28, y0 + ch*cell + 14, f"ch{ch}", 10, "#374151"))
        rows.append(_text(x0 + t*cell + 3, y0 + channels*cell + 16, f"t{t}", 10, "#374151"))


def _arrow(rows: list[str], x1: int, y1: int, x2: int, y2: int, label: str) -> None:
    rows.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#111827" stroke-width="2" marker-end="url(#arrow)"/>')
    rows.append(_text((x1 + x2) / 2 - 28, (y1 + y2) / 2 - 8, label, 12, "#111827"))


def _svg_open(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#111827"/></marker></defs>'
    )


def _text(x: float, y: float, text: str, size: int, color: str = "#111827", weight: str = "normal") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{html.escape(text)}</text>'


def _analysis_md(summary: dict[str, Any], paths: dict[str, Path]) -> str:
    metrics = summary["metrics_by_model"]
    ac = summary["autocorrelation_summary"]
    lines = [
        "# EEG v1 Ridge Baseline Visual Sanity Check",
        "",
        "This is not a new benchmark. It visualizes the existing EEG v1 future-window benchmark so the ridge result is easier to inspect.",
        "",
        f"- dataset: `{summary['dataset']}`",
        f"- source: `{summary['source']}`",
        f"- benchmark_status: `{summary['benchmark_status']}`",
        f"- window_length: `{summary['window_length']}`",
        f"- forecast_horizon: `{summary['forecast_horizon']}`",
        f"- ridge feature matrix: `{summary['ridge_feature_shape']}` after flattening input windows",
        f"- ridge target matrix: `{summary['ridge_target_shape']}` after flattening future windows",
        f"- same-record shifted overlap correlation for the plotted test window: `{summary['same_record_overlap_corr']}`",
        "",
        "## Diagrams",
        "",
        f"![Raw EEG input and target window]({paths['waveform'].name})",
        "",
        f"![Ridge feature map]({paths['feature_map'].name})",
        "",
        f"![Prediction overlay]({paths['prediction_overlay'].name})",
        "",
        "## Current Benchmark Metrics",
        "",
        "| model | mse | mae | r2 | pearsonr |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_id, row in sorted(metrics.items(), key=lambda item: item[1]["mse"]):
        lines.append(f"| {model_id} | {row['mse']:.6g} | {row['mae']:.6g} | {row['r2']:.6g} | {row['pearsonr']:.6g} |")
    lines += [
        "",
        "## Why Ridge Can Look Strong Here",
        "",
        "- `linear_ridge` sees the entire raw EEG input window flattened into scalar time-channel features.",
        "- The target is the future EEG window, not a distant label. With `forecast_horizon=1`, most of the target window is the input window shifted by one sample.",
        "- The synthetic fixture is deliberately smooth/autoregressive, so a linear model is well matched to the data-generating structure.",
        "- Strong ridge or persistence performance is therefore a sanity-check signal for autocorrelation, short horizon, and overlap; it is not evidence of brain-state understanding.",
        "",
        "## Existing Autocorrelation Controls",
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
