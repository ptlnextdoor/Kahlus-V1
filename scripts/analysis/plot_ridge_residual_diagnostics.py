#!/usr/bin/env python3
"""Companion figure: where ridge's h=1 prediction actually fails.

Closes the content gap left when the old fig5 (PSD/residual diagnostics) was
dropped in favor of the single overlap-headline composite: that composite
never asked "how good is the h=1 fit, spectrally and per-channel." This does.

Usage:
    python scripts/analysis/plot_ridge_residual_diagnostics.py \
        --npz artifacts/ridge_bnci_real/ridge_bnci_tensors.npz \
        --out artifacts/ridge_bnci_real/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

# Same validated pair as the headline figure: `node scripts/.../validate_palette.js
# "#2a78d6,#eda100" --mode light` -> ALL CHECKS PASS. Residual is drawn as a neutral
# filled diagnostic, not a 3rd categorical hue -- a flat gray fails the chroma-floor
# check as a categorical slot, which is correct: it isn't an identity-carrying series.
ACTUAL = "#2a78d6"
PRED = "#eda100"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
GRID = "#e6e5e1"
RESIDUAL_FILL = "#c8c7c2"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 400,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.titlesize": 7.8,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.6,
            "axes.edgecolor": INK_3,
            "axes.linewidth": 0.6,
            "axes.labelcolor": INK_2,
            "text.color": INK,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def soft_grid(ax, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def panel_a(ax, y_true, y_pred, sfreq) -> None:
    """PSD of actual vs prediction, with the residual spectrum as a filled diagnostic."""
    flat_true = y_true.reshape(-1, y_true.shape[-1])
    flat_pred = y_pred.reshape(-1, y_pred.shape[-1])
    residual = flat_true - flat_pred

    nperseg = min(256, flat_true.shape[0])
    f_t, pxx_t = welch(flat_true, fs=sfreq, axis=0, nperseg=nperseg)
    f_p, pxx_p = welch(flat_pred, fs=sfreq, axis=0, nperseg=nperseg)
    f_r, pxx_r = welch(residual, fs=sfreq, axis=0, nperseg=nperseg)

    xmax = min(45.0, sfreq / 2)
    ax.fill_between(f_r, np.mean(pxx_r, axis=1), color=RESIDUAL_FILL, alpha=0.55,
                     lw=0, zorder=1, label="residual (actual - prediction)")
    ax.plot(f_t, np.mean(pxx_t, axis=1), color=ACTUAL, lw=1.8, zorder=3, label="actual")
    ax.plot(f_p, np.mean(pxx_p, axis=1), color=PRED, lw=1.4, ls=(0, (3.5, 2)), zorder=4,
            label="ridge prediction")

    ax.set_yscale("log")
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (a.u./Hz, log)")
    ax.set_title("a  h=1 prediction spectrum vs residual", loc="left", fontweight="bold")
    ax.legend(frameon=True, loc="upper right", fontsize=6.2)
    soft_grid(ax)


def panel_b(ax, y_true, y_pred, channel_names) -> None:
    """Per-channel Pearson r, sorted -- single axis, no MSE twin-axis."""
    n = y_true.shape[-1]
    r = np.full(n, np.nan)
    for c in range(n):
        a = y_true[:, :, c].reshape(-1)
        b = y_pred[:, :, c].reshape(-1)
        if np.std(a) > 1e-12 and np.std(b) > 1e-12:
            r[c] = np.corrcoef(a, b)[0, 1]

    order = np.argsort(r)
    names = [channel_names[i] for i in order]
    vals = r[order]

    # All channels sit in a narrow, uniformly-high band (see caption) -- a binary
    # good/bad color split would assert a categorical distinction the data doesn't
    # have. One hue throughout is honest; the sort order already shows the ranking.
    ax.barh(np.arange(n), vals, color=ACTUAL, height=0.68, zorder=3)
    ax.axvline(0, color=INK_3, lw=0.6, zorder=2)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(names, fontsize=5.8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Pearson r  (h=1, per channel)")
    ax.set_title("b  Per-channel fit, sorted", loc="left", fontweight="bold")
    ax.text(0.98, 0.03, f"median $r$={np.nanmedian(r):.2f}\nrange [{np.nanmin(r):.2f}, {np.nanmax(r):.2f}]",
            transform=ax.transAxes, fontsize=6.2, color=INK_2, ha="right", va="bottom")
    soft_grid(ax, axis="x")


CAPTION = r"""\caption{\textbf{Where the h=1 ridge fit actually fails.} \textbf{(a)} Welch power
spectrum (mean over test windows/channels) of the actual signal, the ridge prediction, and their
residual. Prediction tracks actual closely below %(xmax).0f~Hz; residual power is broadband
rather than concentrated in a frequency band the model systematically misses, consistent with
unmodelled noise rather than a structured signal ridge failed to fit. \textbf{(b)} Per-channel
Pearson $r$ between actual and predicted $h{=}1$ signal, sorted; median $r{=}%(med).2f$
(min %(mn).2f, max %(mx).2f). No channel is a qualitative outlier, so the h=1 fit quality is
roughly uniform across scalp location rather than concentrated in a subset of electrodes.}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    style()
    d = np.load(args.npz, allow_pickle=True)
    if "y_pred_test" not in d:
        raise SystemExit("npz has no y_pred_test; rerun build_bnci_ridge_tensors.py")

    y_true = d["y_test"]
    y_pred = d["y_pred_test"]
    sfreq = float(d["sfreq"])
    channel_names = [str(c) for c in d["channel_names"]]

    fig = plt.figure(figsize=(7.0, 3.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])
    panel_a(fig.add_subplot(gs[0, 0]), y_true, y_pred, sfreq)
    panel_b(fig.add_subplot(gs[0, 1]), y_true, y_pred, channel_names)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / "fig_ridge_residual_diagnostics"
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png")
    plt.close(fig)

    flat_true = y_true.reshape(-1, y_true.shape[-1])
    flat_pred = y_pred.reshape(-1, y_pred.shape[-1])
    r_all = np.array([
        np.corrcoef(flat_true[:, c], flat_pred[:, c])[0, 1]
        for c in range(flat_true.shape[-1])
        if np.std(flat_true[:, c]) > 1e-12 and np.std(flat_pred[:, c]) > 1e-12
    ])
    (args.out / "fig_ridge_residual_diagnostics_caption.tex").write_text(
        CAPTION % {"xmax": min(45.0, sfreq / 2), "med": float(np.median(r_all)),
                   "mn": float(np.min(r_all)), "mx": float(np.max(r_all))}
    )
    print(f"wrote {stem}.pdf / .png  (median per-channel r={np.median(r_all):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
