#!/usr/bin/env python3
"""Publication figure: the BNCI2014_001 ridge headline is input/target overlap.

Builds a single 4-panel composite sized for a NeurIPS column. Panels are titled
descriptively; the interpretation belongs in the LaTeX caption (emitted to
caption.tex), not burned into the raster.

Usage:
    python scripts/analysis/plot_ridge_paper_figure.py \
        --npz artifacts/ridge_bnci_real/ridge_bnci_tensors.npz \
        --summary artifacts/ridge_bnci_real/ridge_bnci_summary.json \
        --out artifacts/ridge_bnci_real/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from neurotwin.models.baselines import NumpyRidgeBaseline

# Validated palette (dataviz skill, light surface #fcfcfb).
# `node scripts/validate_palette.js "#2a78d6,#eda100" --mode light` -> ALL CHECKS PASS
# CVD worst adjacent dE 123.7 protan / 80.7 tritan. Contrast WARN on both slots is
# relieved by direct labels on every series (no color-alone identity).
RIDGE = "#2a78d6"   # categorical slot 1
PERSIST = "#eda100"  # categorical slot 3
DIV_LOW = "#2a78d6"  # diverging cool pole
DIV_MID = "#f0efec"  # neutral gray midpoint
DIV_HIGH = "#d03b3b"  # diverging warm pole
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
GRID = "#e6e5e1"

DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", [DIV_LOW, DIV_MID, DIV_HIGH])


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
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,  # editable/embedded type-42, required by most venues
            "ps.fonttype": 42,
        }
    )


def soft_grid(ax, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def detect_lag(x: np.ndarray, y: np.ndarray, probe: int = 8) -> int | None:
    """Recover h from ``Y = signal[s+h : s+h+L]`` by exact-matching y back onto x."""
    length = x.shape[1]
    for cand in range(1, length):
        if np.array_equal(x[:probe, cand:, :], y[:probe, : length - cand, :]):
            return cand
    return None


def fit_ridge_coef(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> np.ndarray:
    """Refit the repo's own ridge baseline so the coefficients are the real ones."""
    model = NumpyRidgeBaseline(alpha=alpha)
    flat_x = x_train.reshape(-1, x_train.shape[-1])
    flat_y = y_train.reshape(-1, y_train.shape[-1])
    model.fit(flat_x, flat_y)
    return np.asarray(model.coef_, dtype=np.float64)


def panel_a(ax, x, y, lag, sfreq, channel_names) -> None:
    """Task construction: the target is the input shifted by h samples."""
    n_show = 15
    ch = 0
    lag_ms = 1000.0 * lag / sfreq
    t_in = np.arange(n_show) * 1000.0 / sfreq
    t_tg = (np.arange(n_show) + lag) * 1000.0 / sfreq
    xi = x[0, :n_show, ch]
    yi = y[0, :n_show, ch]

    # Target drawn ON TOP as a dashed thin line: it must be visible tracing the same
    # path as the input, since "they are the same samples" is the whole point.
    ax.plot(t_in, xi, color=RIDGE, lw=3.0, marker="o", ms=5.0, mew=0, zorder=2,
            solid_capstyle="round", alpha=0.85)
    ax.plot(t_tg, yi, color=PERSIST, lw=1.4, ls=(0, (3.2, 2.2)), marker="o", ms=2.4, mew=0,
            zorder=4, dash_capstyle="round")

    ax.text(0.03, 0.94, "input $X_t$", transform=ax.transAxes, color=RIDGE, fontsize=7.0,
            fontweight="bold", va="top")
    ax.text(0.03, 0.845, f"target $Y=X_{{t+h}}$  (dashed, shifted {lag_ms:.0f} ms)",
            transform=ax.transAxes, color="#8a5c00", fontsize=7.0, fontweight="bold", va="top")

    # Mark the single sample of genuinely new information.
    ax.axvspan(t_in[-1], t_tg[-1], color=GRID, alpha=0.9, zorder=0, lw=0)
    ax.annotate(f"only {lag} new\nsample", xy=(t_tg[-1], yi[-1]), xytext=(-6, -30),
                textcoords="offset points", fontsize=6.0, color=INK_2, ha="right",
                arrowprops=dict(arrowstyle="->", color=INK_2, lw=0.7))

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (z)")
    ax.set_title(f"a  Task construction ({channel_names[ch]}, first {n_show} samples)", loc="left",
                 fontweight="bold")
    soft_grid(ax)


def panel_b(ax, sweep) -> None:
    """Skill vs horizon, log-x so the sub-100 ms action is legible."""
    ms = np.array([r["horizon_ms"] for r in sweep])
    ridge = np.array([r["ridge"]["pearsonr"] for r in sweep])
    pers = np.array([r["persistence"]["pearsonr"] for r in sweep])

    ax.axhspan(-0.12, 0.12, color=GRID, alpha=0.75, zorder=0, lw=0)
    ax.text(560, 0.135, "noise floor", fontsize=6.0, color=INK_3, ha="right")
    ax.axhline(0, color=INK_3, lw=0.5, ls=(0, (3, 3)), zorder=1)

    ax.plot(ms, ridge, color=RIDGE, lw=1.8, marker="o", ms=4.2, mew=0, zorder=4)
    ax.plot(ms, pers, color=PERSIST, lw=1.8, marker="s", ms=4.0, mew=0, zorder=3,
            ls=(0, (4, 2)))

    # Direct labels are the required relief for the palette's contrast WARN, so they
    # must stay clear of the marks they name.
    ax.text(ms[0] * 1.9, 0.92, "ridge", color=RIDGE, fontsize=7.0, fontweight="bold")
    ax.text(ms[0] * 1.9, 0.66, "persistence", color="#8a5c00", fontsize=7.0, fontweight="bold")
    ax.annotate("", xy=(ms[1], pers[1]), xytext=(ms[0] * 3.5, 0.68),
                arrowprops=dict(arrowstyle="-", color=INK_3, lw=0.5))

    ax.set_xscale("log")
    ax.set_xticks([4, 16, 64, 256, 768])
    ax.set_xticklabels(["4", "16", "64", "256", "768"])
    ax.set_xlabel("Forecast horizon (ms, log)")
    ax.set_ylabel("Test Pearson $r$")
    ax.set_ylim(-0.2, 1.0)
    ax.set_title("b  Skill collapses as horizon grows", loc="left", fontweight="bold")
    soft_grid(ax)


def panel_c(ax, sweep) -> None:
    """The money panel: skill plotted directly against overlap (one axis, not two)."""
    ovl = np.array([r["overlap_fraction"] for r in sweep])
    ridge = np.array([r["ridge"]["pearsonr"] for r in sweep])
    hs = [r["horizon_samples"] for r in sweep]

    ax.axhspan(-0.12, 0.12, color=GRID, alpha=0.75, zorder=0, lw=0)
    ax.axhline(0, color=INK_3, lw=0.5, ls=(0, (3, 3)), zorder=1)

    order = np.argsort(ovl)
    ax.plot(ovl[order], ridge[order], color=RIDGE, lw=1.4, zorder=3, alpha=0.55)
    ax.scatter(ovl, ridge, s=46, color=RIDGE, zorder=4, linewidths=0.8, edgecolors="white")

    # h=128 and h=192 both sit at overlap=0; stagger their labels so they cannot collide.
    offsets = {1: (-4, 6, "right"), 4: (-4, 7, "right"), 16: (7, 2, "left"),
               64: (0, -13, "center"), 128: (9, -3, "left"), 192: (9, 4, "left")}
    for o, r_, h in zip(ovl, ridge, hs):
        dx, dy, ha = offsets.get(h, (0, 8, "center"))
        ax.annotate(f"h={h}", (o, r_), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, fontsize=6.0, color=INK_2)

    ax.set_xlabel("Input/target overlap (fraction of $L$=128)")
    ax.set_ylabel("Ridge test Pearson $r$")
    ax.set_ylim(-0.2, 1.0)
    ax.set_xlim(-0.09, 1.12)
    ax.set_title("c  Skill tracks overlap; zero overlap $\\Rightarrow$ noise floor", loc="left",
                 fontweight="bold")
    soft_grid(ax)


def panel_d(ax, coef, channel_names) -> tuple[float, float]:
    """Diagonally dominant, NOT identity.

    mean diagonal beta ~ 0.79 explains why ridge tracks persistence, while the
    remaining off-diagonal mass is the small cross-channel term that lifts ridge
    slightly above it. ||C-I||_F/||I||_F ~ 0.94, so "approximate identity" would
    be an overstatement.
    """
    m = np.abs(coef)
    off = m.copy()
    np.fill_diagonal(off, np.nan)
    mean_diag = float(np.mean(np.diag(coef)))
    ratio = float(np.mean(np.abs(np.diag(coef))) / np.nanmean(off))

    lim = float(np.nanpercentile(np.abs(coef), 99.5)) or 1.0
    im = ax.imshow(coef, cmap=DIVERGING, vmin=-lim, vmax=lim, interpolation="nearest")
    ax.add_patch(Rectangle((-0.5, -0.5), coef.shape[1], coef.shape[0], fill=False,
                           ec=INK_3, lw=0.6))

    n = coef.shape[0]
    ticks = list(range(0, n, 4))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([channel_names[i] for i in ticks], rotation=90, fontsize=5.6)
    ax.set_yticklabels([channel_names[i] for i in ticks], fontsize=5.6)
    ax.set_xlabel("input channel")
    ax.set_ylabel("predicted channel")
    ax.set_title("d  Learned map is diagonally dominant", loc="left", fontweight="bold")

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor(INK_3)
    cb.ax.tick_params(labelsize=5.8, width=0.5, length=2)
    cb.set_label("ridge coefficient", fontsize=6.2)

    ax.text(0.0, -0.32, f"mean diag $\\beta$={mean_diag:.2f} = {ratio:.1f}$\\times$ mean |off-diag|",
            transform=ax.transAxes, fontsize=6.2, color=INK_2)
    return mean_diag, ratio


CAPTION = r"""\caption{\textbf{The ridge headline on BNCI2014\_001 measures input/target overlap, not
neural forecasting.} \textbf{(a)} The benchmark builds $X=s[t{:}t{+}L]$ and $Y=s[t{+}h{:}t{+}h{+}L]$
with $L{=}128$. At the headline setting $h{=}1$ the target is the input shifted by one sample
(4~ms): %(overlap)d of %(L)d samples are bit-identical, so the reported ``forecast'' is
%(frac).1f\%% copied input. \textbf{(b)} Test $r$ for ridge and a matched persistence baseline
both collapse toward the noise floor as $h$ grows, and ridge never separates far from
persistence --- the headline $r{=}0.87$ reflects waveform continuity, not learned dynamics.
\textbf{(c)} Plotting $r$ directly against overlap (rather than against $h$) shows skill tracking
overlap monotonically down to $r\approx0$; once the target no longer overlaps the input
($h\ge L$) ridge sits at $r\approx%(zero).2f$, within the noise floor (the slight non-monotone
tail at zero overlap is noise-floor scatter, not recovered signal). \textbf{(d)} The fitted $22\times22$ channel map is
\emph{diagonally dominant but not an identity}: mean diagonal $\beta{=}%(diag).2f$ is
%(ratio).1f$\times$ the mean $|$off-diagonal$|$ (and $\|C-I\|_F/\|I\|_F{=}%(fro).2f$). The strong
diagonal is why ridge tracks persistence; the residual off-diagonal mass is the small
cross-channel term by which it exceeds persistence at short $h$. Split: subjects 1--6 train,
7--9 test, per-subject $z$-scoring; no cross-subject normalisation.}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    style()
    d = np.load(args.npz, allow_pickle=True)
    summary = json.loads(args.summary.read_text())
    sweep = summary["horizon_sweep"]
    alpha = float(summary.get("ridge_alpha", 1e-2))

    x_test, y_test = d["x_test"], d["y_test"]
    x_train, y_train = d["x_train"], d["y_train"]
    sfreq = float(d["sfreq"])
    channel_names = [str(c) for c in d["channel_names"]]

    lag = detect_lag(x_test, y_test)
    if lag is None:
        raise SystemExit("could not recover the input/target lag from the tensors")
    coef = fit_ridge_coef(x_train, y_train, alpha)

    fig = plt.figure(figsize=(7.0, 4.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1.25, 1.0])
    panel_a(fig.add_subplot(gs[0, 0]), x_test, y_test, lag, sfreq, channel_names)
    panel_b(fig.add_subplot(gs[0, 1]), sweep)
    panel_c(fig.add_subplot(gs[1, 0]), sweep)
    mean_diag, ratio = panel_d(fig.add_subplot(gs[1, 1]), coef, channel_names)
    fro = float(np.linalg.norm(coef - np.eye(coef.shape[0])) / np.linalg.norm(np.eye(coef.shape[0])))

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / "fig_ridge_overlap_headline"
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png")
    plt.close(fig)

    length = x_test.shape[1]
    zero = min(r["ridge"]["pearsonr"] for r in sweep if r["overlap_fraction"] == 0.0)
    (args.out / "fig_ridge_overlap_headline_caption.tex").write_text(
        CAPTION % {"overlap": length - lag, "L": length,
                   "frac": 100.0 * (length - lag) / length,
                   "zero": zero, "diag": mean_diag, "ratio": ratio, "fro": fro}
    )
    print(f"wrote {stem}.pdf / .png  (lag={lag}, mean diag beta={mean_diag:.2f}, "
          f"{ratio:.1f}x off-diag, ||C-I||/||I||={fro:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
