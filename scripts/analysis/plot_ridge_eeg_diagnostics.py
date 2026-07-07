#!/usr/bin/env python3
"""Generate mentor-facing EEG ridge diagnostics.

Scientific contract:
- Real evidence figures must be generated from benchmark tensors, not prompted art.
- The current repo baseline `linear_ridge` fits `NumpyRidgeBaseline(alpha=1e-2)` on
  `_flatten_time(x) == x.reshape(-1, channels)`, i.e. rows are window-time samples
  and columns are channels/features. It is a channel-to-channel linear map, not a
  full flattened time-by-channel window model.
- If no --npz is provided, figures are schematic layout demos and are stamped as
  schematic, not benchmark evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"matplotlib is required: {exc}")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from neurotwin.models.baselines import NumpyRidgeBaseline  # noqa: E402

BLUE = "#2563eb"
RED = "#dc2626"
BLACK = "#111827"
GRAY = "#6b7280"
LIGHT_GRAY = "#e5e7eb"
ORANGE = "#f97316"
PURPLE = "#7c3aed"


@dataclass(frozen=True)
class RidgeFigureData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    y_pred_test: np.ndarray
    sfreq: float
    channel_names: list[str]
    schematic: bool
    prediction_source: str
    provenance: dict[str, str]
    coef: np.ndarray


def apply_publication_style() -> None:
    """Use paper-safe matplotlib defaults while avoiding fragile LaTeX deps."""
    try:  # SciencePlots is nice when present, but never required.
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
    except Exception:
        plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _as_3d(a: np.ndarray, time_length: int | None, n_channels: int | None, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 3:
        out = a
    elif a.ndim == 2:
        if time_length is None or n_channels is None:
            raise ValueError(f"{name}: flattened arrays require --time-length and --n-channels")
        if a.shape[1] != time_length * n_channels:
            raise ValueError(f"{name}: feature dim {a.shape[1]} != {time_length}*{n_channels}")
        out = a.reshape(a.shape[0], time_length, n_channels)
    else:
        raise ValueError(f"{name}: expected 2D or 3D array, got {a.shape}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name}: contains NaN or Inf")
    return out


def _flatten_time(a: np.ndarray) -> np.ndarray:
    """Match `baseline_suite._flatten_time`: rows = samples*time, cols = channels."""
    return np.asarray(a, dtype=np.float64).reshape(-1, a.shape[-1])


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or b.size < 2 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _demo_data(seed: int = 0, n_train: int = 320, n_test: int = 80, t: int = 128, c: int = 8):
    """Generate EEG-like schematic data. This is never benchmark evidence."""
    rng = np.random.default_rng(seed)
    total_windows = n_train + n_test + 1
    signal = np.zeros((total_windows, t, c), dtype=np.float64)
    phase = rng.uniform(0, 2 * np.pi, size=c)
    freqs = np.linspace(6, 22, c)
    time = np.arange(total_windows * t) / 128.0
    raw = np.stack([np.sin(2 * np.pi * f * time + ph) for f, ph in zip(freqs, phase)], axis=1)
    raw += 0.35 * rng.standard_normal(raw.shape)
    mix = np.eye(c) * 0.85 + rng.normal(0, 0.04, size=(c, c))
    raw = np.einsum("tc,cd->td", raw, mix)
    kernel = np.array([0.15, 0.70, 0.15])
    for ch in range(c):
        raw[:, ch] = np.convolve(raw[:, ch], kernel, mode="same")
    for i in range(total_windows):
        signal[i] = raw[i * t : (i + 1) * t]
    x = signal[:-1]
    y = signal[1:]
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:], 128.0, [f"Ch {i+1}" for i in range(c)]


def _npz_strings(data: np.lib.npyio.NpzFile, key: str, fallback: list[str]) -> list[str]:
    if key not in data:
        return fallback
    arr = data[key]
    if arr.ndim == 0:
        return [str(arr.item())]
    return [str(x) for x in arr.tolist()]


def load_data(args: argparse.Namespace) -> RidgeFigureData:
    provenance: dict[str, str] = {}
    if args.npz is None:
        x_train, y_train, x_test, y_test, sfreq, channel_names = _demo_data(args.seed)
        schematic = True
        provenance["source"] = "synthetic schematic generated by plotting script"
    else:
        data = np.load(args.npz, allow_pickle=True)
        x_train = _as_3d(data["x_train"], args.time_length, args.n_channels, "x_train")
        y_train = _as_3d(data["y_train"], args.time_length, args.n_channels, "y_train")
        x_test = _as_3d(data["x_test"], args.time_length, args.n_channels, "x_test")
        y_test = _as_3d(data["y_test"], args.time_length, args.n_channels, "y_test")
        sfreq = float(data["sfreq"]) if "sfreq" in data else float(args.sfreq)
        channel_names = _npz_strings(data, "channel_names", [f"Ch {i+1}" for i in range(x_train.shape[-1])])
        schematic = False
        provenance["source"] = str(args.npz)
        for key in ("dataset", "task_id", "split_manifest", "event_manifest", "commit", "run_label"):
            if key in data:
                value = data[key]
                provenance[key] = str(value.item() if getattr(value, "ndim", 1) == 0 else value.tolist())
        if "y_pred_test" in data:
            y_pred_test = _as_3d(data["y_pred_test"], args.time_length, args.n_channels, "y_pred_test")
            if y_pred_test.shape != y_test.shape:
                raise ValueError(f"y_pred_test shape {y_pred_test.shape} != y_test shape {y_test.shape}")
            prediction_source = "provided y_pred_test from benchmark/export artifact"
        else:
            y_pred_test = None
            prediction_source = "recomputed linear_ridge from x_train/y_train using repo NumpyRidgeBaseline alpha=1e-2"

    _validate_arrays(x_train, y_train, x_test, y_test, channel_names, sfreq)
    model = NumpyRidgeBaseline(alpha=args.alpha)
    model.fit(_flatten_time(x_train), _flatten_time(y_train))
    coef = np.asarray(model.coef_, dtype=np.float64)
    if args.npz is None:
        y_pred_test = None
        prediction_source = "schematic recomputed linear_ridge on synthetic demo data"
    if y_pred_test is None:
        pred_flat = model.predict(_flatten_time(x_test))
        y_pred_test = pred_flat.reshape(x_test.shape[0], x_test.shape[1], y_train.shape[-1])
        if y_pred_test.shape != y_test.shape:
            raise ValueError(f"recomputed prediction shape {y_pred_test.shape} != y_test shape {y_test.shape}")
    return RidgeFigureData(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        y_pred_test=y_pred_test,
        sfreq=sfreq,
        channel_names=channel_names,
        schematic=schematic,
        prediction_source=prediction_source,
        provenance=provenance,
        coef=coef,
    )


def _validate_arrays(x_train, y_train, x_test, y_test, channel_names: list[str], sfreq: float) -> None:
    shapes = {"x_train": x_train.shape, "y_train": y_train.shape, "x_test": x_test.shape, "y_test": y_test.shape}
    for name, arr in (("x_train", x_train), ("y_train", y_train), ("x_test", x_test), ("y_test", y_test)):
        if arr.ndim != 3:
            raise ValueError(f"{name} must be [windows, time, channels], got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN or Inf")
    if x_train.shape[1] != y_train.shape[1] or x_test.shape[1] != y_test.shape[1]:
        raise ValueError(f"input/target time lengths must match for current linear_ridge diagnostic: {shapes}")
    if x_train.shape[-1] != y_train.shape[-1] or x_test.shape[-1] != y_test.shape[-1]:
        raise ValueError(f"input/target channel counts must match for EEG->EEG ridge diagnostic: {shapes}")
    if x_train.shape[-1] != x_test.shape[-1] or y_train.shape[-1] != y_test.shape[-1]:
        raise ValueError(f"train/test channel counts disagree: {shapes}")
    if len(channel_names) != x_train.shape[-1]:
        raise ValueError(f"channel_names length {len(channel_names)} != n_channels {x_train.shape[-1]}")
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError(f"sfreq must be positive and finite, got {sfreq}")


def selected_channels(names: list[str], max_channels: int, requested: str | None) -> list[int]:
    if requested:
        out: list[int] = []
        lower = {name.lower(): i for i, name in enumerate(names)}
        for raw in requested.split(","):
            token = raw.strip()
            if not token:
                continue
            if token.isdigit():
                idx = int(token)
                if idx >= len(names):
                    raise ValueError(f"channel index {idx} out of range")
                out.append(idx)
            else:
                key = token.lower()
                if key not in lower:
                    raise ValueError(f"unknown channel '{token}'")
                out.append(lower[key])
        return list(dict.fromkeys(out))[:max_channels]
    return list(range(min(max_channels, len(names))))


def stamp_axis(ax, schematic: bool, source: str) -> None:
    text = "SCHEMATIC - NOT BENCHMARK EVIDENCE" if schematic else "BENCHMARK-DERIVED FIGURE"
    ax.text(
        0.0,
        -0.22,
        f"{text}. Prediction source: {source}.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color=GRAY,
    )


def save_figure(fig, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.png")
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)


def panel_label(ax, label: str) -> None:
    ax.text(-0.08, 1.06, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left")


def figure_input_target(data: RidgeFigureData, channels: list[int], out: Path) -> None:
    idx = 0
    x, y = data.x_test, data.y_test
    tx = np.arange(x.shape[1]) / data.sfreq
    ty = (np.arange(y.shape[1]) + x.shape[1]) / data.sfreq
    fig = plt.figure(figsize=(11.2, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.4, 1.1])

    ax = fig.add_subplot(gs[0, :])
    panel_label(ax, "A")
    scale = np.nanstd(np.concatenate([x[idx, :, channels], y[idx, :, channels]], axis=0)) or 1.0
    for row, ch in enumerate(channels):
        offset = row * 2.6
        ax.plot(tx, x[idx, :, ch] / scale + offset, color=BLUE, lw=1.05)
        ax.plot(ty, y[idx, :, ch] / scale + offset, color=RED, lw=1.05)
        ax.text(-0.03, offset, data.channel_names[ch], va="center", ha="right", fontsize=8)
    ax.axvspan(tx[0], tx[-1], color=BLUE, alpha=0.08, label="input window X_t")
    ax.axvspan(ty[0], ty[-1], color=RED, alpha=0.08, label="target next window X_{t+1}")
    ax.axvline(tx[-1], color=BLACK, lw=1, ls="--", alpha=0.65)
    ax.set_title("Raw EEG example: current window passed to ridge and next-window target")
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.legend(loc="upper right", frameon=True)

    ax2 = fig.add_subplot(gs[1, 0])
    panel_label(ax2, "B")
    vmax = np.nanpercentile(np.abs(x[idx, :, channels]), 98) or 1.0
    im = ax2.imshow(x[idx, :, channels].T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax2.set_title("One input window: time x selected channels")
    ax2.set_xlabel("time sample")
    ax2.set_ylabel("channel")
    ax2.set_yticks(range(len(channels)), [data.channel_names[ch] for ch in channels])
    fig.colorbar(im, ax=ax2, fraction=0.046, label="amplitude")

    ax3 = fig.add_subplot(gs[1, 1])
    panel_label(ax3, "C")
    matrix = _flatten_time(data.x_train[: min(30, data.x_train.shape[0]), :, :][:, :, channels])
    vmax = np.nanpercentile(np.abs(matrix), 98) or 1.0
    ax3.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax3.set_title("Ridge design matrix subset")
    ax3.set_xlabel("features = channels")
    ax3.set_ylabel("rows = train windows x time samples")
    ax3.set_xticks(range(len(channels)), [data.channel_names[ch] for ch in channels], rotation=45, ha="right")
    stamp_axis(ax3, data.schematic, data.prediction_source)

    save_figure(fig, out, "fig1_ridge_input_target_waveforms")


def per_channel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_channels = y_true.shape[-1]
    corr = np.empty(n_channels, dtype=np.float64)
    mse = np.empty(n_channels, dtype=np.float64)
    for ch in range(n_channels):
        corr[ch] = _safe_corr(y_true[:, :, ch], y_pred[:, :, ch])
        mse[ch] = float(np.mean((y_true[:, :, ch] - y_pred[:, :, ch]) ** 2))
    return corr, mse


def figure_prediction(data: RidgeFigureData, channels: list[int], out: Path) -> None:
    idx = 0
    y_true = data.y_test
    y_pred = data.y_pred_test
    t = np.arange(y_true.shape[1]) / data.sfreq
    fig, axes = plt.subplots(len(channels), 1, figsize=(11.2, 1.35 * len(channels) + 1.3), sharex=True, constrained_layout=True)
    if len(channels) == 1:
        axes = [axes]
    for ax, ch in zip(axes, channels):
        yt = y_true[idx, :, ch]
        yp = y_pred[idx, :, ch]
        r = _safe_corr(yt, yp)
        mse = float(np.mean((yt - yp) ** 2))
        ax.plot(t, yt, color=BLACK, lw=1.25, label="actual future EEG")
        ax.plot(t, yp, color=RED, lw=1.05, alpha=0.88, label="ridge prediction")
        ax.plot(t, yt - yp, color=GRAY, lw=0.80, alpha=0.75, label="residual")
        ax.axhline(0, color=LIGHT_GRAY, lw=0.7)
        ax.set_ylabel(data.channel_names[ch], rotation=0, ha="right", va="center")
        ax.text(0.995, 0.76, f"r={r:.2f}, MSE={mse:.2g}", transform=ax.transAxes, ha="right", fontsize=8)
    panel_label(axes[0], "A")
    axes[0].legend(loc="upper left", ncol=3, frameon=True)
    axes[0].set_title("Ridge prediction overlay: next-window EEG")
    axes[-1].set_xlabel("Time into predicted window (s)")
    stamp_axis(axes[-1], data.schematic, data.prediction_source)
    save_figure(fig, out, "fig2_ridge_prediction_overlay")


def figure_autocorr(data: RidgeFigureData, channels: list[int], out: Path) -> None:
    x, y = data.x_train, data.y_train
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), constrained_layout=True)
    max_lag = min(int(round(data.sfreq * 0.75)), x.shape[1] - 2)
    max_lag = max(2, max_lag)
    lags = np.arange(max_lag + 1) / data.sfreq
    panel_label(axes[0], "A")
    for ch in channels:
        sig = x[:, :, ch].reshape(-1)
        sig = sig - sig.mean()
        denom = float(np.dot(sig, sig)) or 1.0
        ac = [float(np.dot(sig[:-lag or None], sig[lag:]) / denom) for lag in range(max_lag + 1)]
        axes[0].plot(lags, ac, lw=1.05, label=data.channel_names[ch])
    axes[0].set_title("Short-horizon autocorrelation in input EEG")
    axes[0].set_xlabel("Lag (s)")
    axes[0].set_ylabel("autocorrelation")
    axes[0].legend(fontsize=7, ncol=2, frameon=True)

    panel_label(axes[1], "B")
    x_summary = x[:, :, channels].mean(axis=1)
    y_summary = y[:, :, channels].mean(axis=1)
    corr = np.corrcoef(x_summary.T, y_summary.T)[: len(channels), len(channels) :]
    im = axes[1].imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[1].set_title("Current-window vs next-window channel correlation")
    axes[1].set_xticks(range(len(channels)), [data.channel_names[ch] for ch in channels], rotation=45, ha="right")
    axes[1].set_yticks(range(len(channels)), [data.channel_names[ch] for ch in channels])
    fig.colorbar(im, ax=axes[1], fraction=0.046, label="Pearson r")
    stamp_axis(axes[1], data.schematic, data.prediction_source)
    save_figure(fig, out, "fig3_autocorrelation_lag_structure")


def figure_coefficients(data: RidgeFigureData, channels: list[int], out: Path) -> None:
    coef = data.coef[np.ix_(channels, channels)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True)
    panel_label(axes[0], "A")
    vmax = np.nanpercentile(np.abs(coef), 98) or 1.0
    im = axes[0].imshow(coef, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Signed ridge channel-to-channel coefficients")
    axes[0].set_xlabel("predicted/output channel")
    axes[0].set_ylabel("input channel")
    axes[0].set_xticks(range(len(channels)), [data.channel_names[i] for i in channels], rotation=45, ha="right")
    axes[0].set_yticks(range(len(channels)), [data.channel_names[i] for i in channels])
    fig.colorbar(im, ax=axes[0], fraction=0.046, label="weight")

    panel_label(axes[1], "B")
    mag = np.abs(coef)
    im2 = axes[1].imshow(mag, cmap="magma")
    axes[1].set_title("Coefficient magnitude")
    axes[1].set_xlabel("predicted/output channel")
    axes[1].set_ylabel("input channel")
    axes[1].set_xticks(range(len(channels)), [data.channel_names[i] for i in channels], rotation=45, ha="right")
    axes[1].set_yticks(range(len(channels)), [data.channel_names[i] for i in channels])
    fig.colorbar(im2, ax=axes[1], fraction=0.046, label="|weight|")
    stamp_axis(axes[1], data.schematic, data.prediction_source)
    save_figure(fig, out, "fig4_ridge_coefficient_channel_map")


def figure_psd_residual(data: RidgeFigureData, channels: list[int], out: Path) -> None:
    try:
        from scipy.signal import welch
    except Exception:
        return
    y_true = data.y_test[:, :, channels].reshape(-1, len(channels))
    y_pred = data.y_pred_test[:, :, channels].reshape(-1, len(channels))
    residual = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.6), constrained_layout=True)
    panel_label(axes[0], "A")
    for arr, label, color in ((y_true, "actual", BLACK), (y_pred, "ridge prediction", RED), (residual, "residual", GRAY)):
        freqs, pxx = welch(arr, fs=data.sfreq, axis=0, nperseg=min(256, arr.shape[0]))
        axes[0].plot(freqs, np.mean(pxx, axis=1), color=color, lw=1.25, label=label)
    axes[0].set_yscale("log")
    axes[0].set_xlim(0, min(45, data.sfreq / 2))
    axes[0].set_title("Power spectrum: actual vs prediction vs residual")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("PSD (a.u./Hz)")
    axes[0].legend(frameon=True)

    panel_label(axes[1], "B")
    corr, mse = per_channel_metrics(data.y_test, data.y_pred_test)
    shown_corr = corr[channels]
    shown_mse = mse[channels]
    xloc = np.arange(len(channels))
    axes[1].bar(xloc - 0.18, shown_corr, width=0.36, color=BLUE, label="Pearson r")
    ax2 = axes[1].twinx()
    ax2.bar(xloc + 0.18, shown_mse, width=0.36, color=ORANGE, alpha=0.8, label="MSE")
    axes[1].set_title("Per-channel predictive quality")
    axes[1].set_xticks(xloc, [data.channel_names[i] for i in channels], rotation=45, ha="right")
    axes[1].set_ylabel("Pearson r", color=BLUE)
    ax2.set_ylabel("MSE", color=ORANGE)
    axes[1].axhline(0, color=LIGHT_GRAY, lw=0.8)
    lines, labels = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines + lines2, labels + labels2, frameon=True, loc="upper right")
    stamp_axis(axes[1], data.schematic, data.prediction_source)
    save_figure(fig, out, "fig5_psd_residual_diagnostics")


def figure_topomap_if_possible(data: RidgeFigureData, out: Path, montage: str) -> str | None:
    if data.schematic:
        return "skipped topomap: schematic channels do not have valid sensor positions"
    try:
        import mne
    except Exception as exc:
        return f"skipped topomap: MNE unavailable: {exc}"
    generic = all(name.lower().startswith("ch ") for name in data.channel_names)
    if generic:
        return "skipped topomap: generic channel names cannot be mapped to a montage"
    try:
        info = mne.create_info(data.channel_names, data.sfreq, ch_types="eeg")
        info.set_montage(montage, on_missing="ignore")
        corr, mse = per_channel_metrics(data.y_test, data.y_pred_test)
        importance = np.mean(np.abs(data.coef), axis=1)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
        for ax, values, title, cmap in (
            (axes[0], corr, "prediction correlation", "viridis"),
            (axes[1], mse, "prediction MSE", "magma"),
            (axes[2], importance, "input coefficient magnitude", "magma"),
        ):
            mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap=cmap, contours=0)
            ax.set_title(title)
        panel_label(axes[0], "A")
        panel_label(axes[1], "B")
        panel_label(axes[2], "C")
        stamp_axis(axes[2], data.schematic, data.prediction_source)
        save_figure(fig, out, "fig6_mne_sensor_topomaps")
        return None
    except Exception as exc:  # topomaps are optional and data-dependent.
        return f"skipped topomap: {exc}"


def metric_summary(data: RidgeFigureData) -> dict[str, float]:
    return {
        "global_mse": float(np.mean((data.y_test - data.y_pred_test) ** 2)),
        "global_mae": float(np.mean(np.abs(data.y_test - data.y_pred_test))),
        "global_pearsonr": _safe_corr(data.y_test, data.y_pred_test),
        "n_train_windows": float(data.x_train.shape[0]),
        "n_test_windows": float(data.x_test.shape[0]),
        "window_timepoints": float(data.x_train.shape[1]),
        "n_channels": float(data.x_train.shape[2]),
        "sfreq": float(data.sfreq),
    }


def write_readme(out: Path, data: RidgeFigureData, skipped: Iterable[str]) -> None:
    status = "schematic demo figures, not benchmark evidence" if data.schematic else "generated from benchmark/export tensors"
    metrics = metric_summary(data)
    files = sorted(p.name for p in out.glob("fig*.png"))
    provenance_lines = "\n".join(f"- `{k}`: `{v}`" for k, v in sorted(data.provenance.items())) or "- none recorded"
    metric_lines = "\n".join(f"- `{k}`: {v:.6g}" for k, v in metrics.items())
    skipped_lines = "\n".join(f"- {s}" for s in skipped) if skipped else "- none"
    out.joinpath("README.md").write_text(
        "# Ridge EEG diagnostic figures\n\n"
        f"**Status:** {status}.\n\n"
        f"**Prediction source:** {data.prediction_source}.\n\n"
        "## Critical interpretation note\n\n"
        "The repo's current `linear_ridge` baseline fits a ridge model after reshaping "
        "`[windows, time, channels]` into `[windows*time, channels]`. It therefore learns a regularized "
        "channel-to-channel linear map, not a full time-by-channel flattened-window model. "
        "High performance should be interpreted as evidence that near-future EEG is linearly predictable "
        "from recent channel covariance/temporal continuity under the current split, not as proof of a rich neural state model.\n\n"
        "## Generated files\n\n"
        + "\n".join(f"- `{name}`" for name in files)
        + "\n\n## Metrics from plotted prediction\n\n"
        + metric_lines
        + "\n\n## Provenance\n\n"
        + provenance_lines
        + "\n\n## Optional outputs skipped\n\n"
        + skipped_lines
        + "\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, help="Benchmark tensor npz with x_train/y_train/x_test/y_test and optional y_pred_test")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "analysis" / "ridge_eeg_figures")
    ap.add_argument("--time-length", type=int, help="Required only when arrays are flattened")
    ap.add_argument("--n-channels", type=int, help="Required only when arrays are flattened")
    ap.add_argument("--sfreq", type=float, default=128.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1e-2, help="Must match baseline_suite linear_ridge alpha unless intentionally changed")
    ap.add_argument("--max-display-channels", type=int, default=8)
    ap.add_argument("--channels", type=str, help="Comma-separated channel names or zero-based indices to display")
    ap.add_argument("--montage", type=str, default="standard_1020")
    args = ap.parse_args()

    apply_publication_style()
    args.out.mkdir(parents=True, exist_ok=True)
    data = load_data(args)
    channels = selected_channels(data.channel_names, args.max_display_channels, args.channels)

    figure_input_target(data, channels, args.out)
    figure_prediction(data, channels, args.out)
    figure_autocorr(data, channels, args.out)
    figure_coefficients(data, channels, args.out)
    figure_psd_residual(data, channels, args.out)
    skipped = []
    topomap_skip = figure_topomap_if_possible(data, args.out, args.montage)
    if topomap_skip:
        skipped.append(topomap_skip)
    write_readme(args.out, data, skipped)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
