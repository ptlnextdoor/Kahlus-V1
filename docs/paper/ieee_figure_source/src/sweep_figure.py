"""kahlus-bench sweep figure helper.

Reproduces the look of kahlus_bench/sweep.py::_write_figure — a clean multi-panel
matplotlib comparison of several methods across a swept parameter, with a black
dashed analytic-truth reference, per-method colored marker lines, direction-
carrying panel titles, and a one-sentence suptitle.

Usage: see SKILL.md, or run this file directly for a self-contained demo that
writes sweep_figure_demo.png.
"""
from __future__ import annotations


def sweep_figure(rows, x, panels, out, x_label=None, suptitle=None,
                 method_key="method", figsize=None, dpi=130):
    """Render the sweep figure.

    rows      : list[dict], one dict per (method, x-value) with metric keys.
    x         : key in each row for the swept parameter (x-axis).
    panels    : list[dict], each with:
                  y      : metric key to plot
                  label  : y-axis label (full phrase)
                  title  : panel title (carry direction, e.g. "lower is better")
                  ylim   : optional (lo, hi); hi may be None to autoscale up
                  truth  : optional key for a black dashed reference line
    out       : output image path.
    x_label   : shared x-axis label.
    suptitle  : one-sentence takeaway across the top.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = sorted({r[method_key] for r in rows})
    n = len(panels)
    if figsize is None:
        figsize = (5.5 * n, 4.5)
    fig, axes = plt.subplots(1, n, figsize=figsize, constrained_layout=True)
    if n == 1:
        axes = [axes]

    for pi, panel in enumerate(panels):
        ax = axes[pi]
        truth_key = panel.get("truth")
        for mi, name in enumerate(methods):
            sub = sorted([r for r in rows if r[method_key] == name],
                         key=lambda r: r[x])
            if not sub:
                continue
            xs = [r[x] for r in sub]
            if truth_key and mi == 0:
                ax.plot(xs, [r[truth_key] for r in sub], color="black",
                        linestyle="--", linewidth=1.0, label="truth")
            ax.plot(xs, [r[panel["y"]] for r in sub], marker="o",
                    markersize=4, linewidth=1.5, label=name)
        ax.set_xlabel(x_label or x)
        ax.set_ylabel(panel["label"])
        ax.set_title(panel["title"])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ylim = panel.get("ylim")
        if ylim is not None:
            lo, hi = ylim
            if hi is None:
                hi = max(0.3, ax.get_ylim()[1])
            ax.set_ylim(lo, hi)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def _demo():
    import math
    import random
    random.seed(0)
    methods = {"Correlation": 0.26, "PairwiseTE": 0.03, "ConditionalVAR": 0.0}
    grid = [i / 20 for i in range(9)]  # a = 0.0 .. 0.4
    rows = []
    for name, fpr_ceiling in methods.items():
        for a in grid:
            truth = 0.72 * a * a  # closed-form-ish analytic truth in bits
            det = 1 / (1 + math.exp(-40 * (a - 0.12)))  # detection power S-curve
            rows.append(dict(
                method=name, a=a, truth=truth,
                est=truth - 0.02 + random.uniform(-0.01, 0.01),
                det=round(det, 3),
                fpr=round(fpr_ceiling * det, 3),
            ))
    sweep_figure(
        rows, x="a", x_label="true coupling a",
        panels=[
            dict(y="est", label="certified DI estimate (bits)",
                 title="median certified DI vs analytic truth", truth="truth"),
            dict(y="det", label="detection power", ylim=(-0.05, 1.05),
                 title="detection power vs coupling"),
            dict(y="fpr", label="false-positive rate on null edges",
                 ylim=(-0.02, None),
                 title="false-certify rate vs coupling  (lower is better)"),
        ],
        suptitle=("Detection-power sweep: matched detection power, "
                  "divergent false-certify rate"),
        out="sweep_figure_demo.png",
    )
    print("wrote sweep_figure_demo.png")


if __name__ == "__main__":
    _demo()
