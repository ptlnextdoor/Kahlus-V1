"""BNCI2014_001 isolated future-sample forecast check (Amrith protocol).

Run:
  PYTHONPATH=src python3 scripts/bnci_isolated_forecast_check.py --selftest
  PYTHONPATH=src python3 scripts/bnci_isolated_forecast_check.py \\
      --out runs/bnci_isolated_check --subjects 9 --channel C3
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from amrith_isolated_forecast_check import (  # noqa: E402
    HORIZONS,
    make_windows,
    old_overlapping_table,
    predict_all,
    selftest,
)
from analysis.build_bnci_ridge_tensors import (  # noqa: E402
    DEFAULT_CACHE,
    EEG_CHANNEL_NAMES,
    load_subject_eeg,
    zscore_train,
)

SEED = 0
CHANNEL_INDEX = {"C3": 7, "Cz": 9}


def cluster_bootstrap_gap(
    per_subject: dict[int, dict[str, float]],
    *,
    seed: int = SEED,
    n_boot: int = 2000,
) -> dict[str, float]:
    """Bootstrap subject clusters on h=1 gap = best_trivial_mse - gru_mse."""
    subjects = sorted(per_subject)
    gaps = np.asarray(
        [
            min(row["mean_of_trace"], row["persistence"], row["ridge_ar"]) - row["kahlus_gru"]
            for row in (per_subject[s] for s in subjects)
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    samples = [
        float(np.mean(gaps[rng.integers(0, len(gaps), size=len(gaps))]))
        for _ in range(n_boot)
    ]
    ci_low = float(np.percentile(samples, 2.5))
    ci_high = float(np.percentile(samples, 97.5))
    return {
        "observed_gap": float(np.mean(gaps)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bootstrap_samples": n_boot,
        "gru_beats_best_trivial_point_estimate": bool(np.mean(gaps) > 0),
        "gru_beats_best_trivial_ci_excludes_zero": bool(ci_low > 0.0),
    }


def per_subject_h1_mse(
    train_ctx,
    train_tgt,
    test_ctx,
    test_tgt,
    *,
    test_subject_slices: dict[int, slice],
    rng: np.random.Generator,
) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for subj, sl in test_subject_slices.items():
        sweep = predict_all(
            train_ctx,
            train_tgt,
            test_ctx[sl],
            {h: test_tgt[h][sl] for h in HORIZONS},
            rng=rng,
        )
        out[subj] = {m: sweep[m][1] for m in sweep}
    return out


def write_report(
    path: Path,
    *,
    headline: dict,
    bootstrap: dict,
    channel: str,
    channel_mode: str,
    train_subjects: list[int],
    test_subjects: list[int],
) -> None:
    verdict = (
        "FORECASTING_SKILL_SURVIVES"
        if bootstrap["gru_beats_best_trivial_ci_excludes_zero"]
        else "FORECASTING_SKILL_DEAD"
    )
    lines = [
        "# BNCI isolated forecast check",
        "",
        f"- channel: `{channel}` (mode={channel_mode})",
        f"- train subjects: {train_subjects}",
        f"- test subjects: {test_subjects}",
        "",
        "## Decision rule",
        "",
        "GRU beats best trivial baseline at h=1 **and** subject-cluster bootstrap CI on",
        "`best_trivial_mse - gru_mse` excludes 0 => forecasting-skill claim survives.",
        "Otherwise drop the IEEE forecasting win; keep Neural-CASP / copy-trap / overlap audit.",
        "",
        f"## Verdict: **{verdict}**",
        "",
        f"- observed gap (best trivial - GRU): {bootstrap['observed_gap']:.6f}",
        f"- bootstrap 95% CI: [{bootstrap['ci_low']:.6f}, {bootstrap['ci_high']:.6f}]",
        "",
        "## Headline JSON",
        "",
        "```json",
        json.dumps(headline, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=Path("runs/bnci_isolated_check"))
    ap.add_argument("--subjects", type=int, default=9)
    ap.add_argument("--channel", type=str, default="C3")
    ap.add_argument("--channel-mode", choices=["single", "mean"], default="single")
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--max-windows", type=int, default=500)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    selftest()
    rng = np.random.default_rng(SEED)
    subject_ids = list(range(1, args.subjects + 1))
    n_test = max(1, len(subject_ids) // 4)
    test_subjects = set(subject_ids[-n_test:])
    train_subjects = [s for s in subject_ids if s not in test_subjects]

    tr_ctx_parts, tr_tgt_parts = [], {h: [] for h in HORIZONS}
    te_ctx_parts, te_tgt_parts = [], {h: [] for h in HORIZONS}
    te_slices: dict[int, slice] = {}
    offset = 0

    train_stack = []
    for subj in train_subjects:
        try:
            sig = load_subject_eeg(args.cache_dir, subj)[0]
            train_stack.append(sig)
        except FileNotFoundError:
            continue
    if not train_stack:
        ap.error("no BNCI training subjects found in cache")
    train_concat = np.concatenate(train_stack, axis=0)

    for subj in subject_ids:
        try:
            raw = load_subject_eeg(args.cache_dir, subj)[0]
        except FileNotFoundError:
            print(f"skip subject {subj}: missing cache", file=sys.stderr)
            continue
        if args.channel_mode == "mean":
            trace = zscore_train(train_concat, raw.mean(axis=1))
        else:
            idx = CHANNEL_INDEX.get(args.channel, EEG_CHANNEL_NAMES.index(args.channel))
            trace = zscore_train(train_concat[:, idx], raw[:, idx])
        ctx, tgt = make_windows(trace, stride=args.stride, max_windows=args.max_windows, rng=rng)
        if subj in test_subjects:
            te_slices[subj] = slice(offset, offset + len(ctx))
            offset += len(ctx)
            te_ctx_parts.append(ctx)
            for h in HORIZONS:
                te_tgt_parts[h].append(tgt[h])
        else:
            tr_ctx_parts.append(ctx)
            for h in HORIZONS:
                tr_tgt_parts[h].append(tgt[h])

    train_ctx = np.concatenate(tr_ctx_parts)
    test_ctx = np.concatenate(te_ctx_parts)
    train_tgt = {h: np.concatenate(tr_tgt_parts[h]) for h in HORIZONS}
    test_tgt = {h: np.concatenate(te_tgt_parts[h]) for h in HORIZONS}
    print(f"train windows={len(train_ctx)} test windows={len(test_ctx)}")
    print(f"held-out test subjects={sorted(test_subjects)}")

    sweep = predict_all(train_ctx, train_tgt, test_ctx, test_tgt, rng=rng)
    old = old_overlapping_table(train_ctx, train_tgt[1], test_ctx, test_tgt[1])
    per_subj = per_subject_h1_mse(
        train_ctx,
        train_tgt,
        test_ctx,
        test_tgt,
        test_subject_slices=te_slices,
        rng=rng,
    )
    bootstrap = cluster_bootstrap_gap(per_subj, seed=SEED)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = [{"horizon": h, "method": m, "mse": sweep[m][h]} for m in sweep for h in HORIZONS]
    with (args.out / "sweep.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["horizon", "method", "mse"])
        w.writeheader()
        w.writerows(rows)

    headline = {
        "dataset": "BNCI2014_001",
        "channel": args.channel,
        "channel_mode": args.channel_mode,
        "train_subjects": train_subjects,
        "test_subjects": sorted(test_subjects),
        "old_overlapping_metric_full_127_target": old,
        "new_isolated_metric_h1": {m: sweep[m][1] for m in sweep},
        "h1_subject_bootstrap": bootstrap,
        "per_subject_h1_mse": {str(k): v for k, v in per_subj.items()},
        "note": (
            "OLD shares 126/127 samples between input and target; NEW predicts only "
            "the strictly-future sample. Subject-held-out BNCI2014_001."
        ),
    }
    (args.out / "headline.json").write_text(json.dumps(headline, indent=2), encoding="utf-8")
    write_report(
        args.out / "REPORT.md",
        headline=headline,
        bootstrap=bootstrap,
        channel=args.channel,
        channel_mode=args.channel_mode,
        train_subjects=train_subjects,
        test_subjects=sorted(test_subjects),
    )

    print(json.dumps(headline, indent=2))
    print(f"wrote {args.out}/")


if __name__ == "__main__":
    main()
