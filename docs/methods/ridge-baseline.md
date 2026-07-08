# Ridge baseline evidence note

```{admonition} Audience
:class: note
This page is written in two layers: plain explanation first, then equations and implementation details for technical readers.
```

## Plain explanation

Ridge regression is a linear model with a safety brake. It predicts a target from input features while penalizing weights that become too large.

For Kahlus EEG/ridge docs, the most important rule is epistemic honesty: a figure should only show what the saved artifacts can support. If the archive contains metrics and audits, plot metrics and audits. If it does not contain raw tensors or saved predictions, do not draw waveform overlays or prediction traces.

## Formal objective

Let $X \in \mathbb{R}^{n \times d}$ be the input design matrix and $Y \in \mathbb{R}^{n \times m}$ be the target matrix. Ridge solves:

```{math}
\hat{W} = \arg\min_W \lVert XW - Y \rVert_2^2 + \alpha \lVert W \rVert_2^2
```

The evidence figures do not currently reconstruct $X$, $Y$, or $\hat{Y}$ because the versions evidence zips do not contain raw tensor arrays, epoch files, or prediction arrays. Instead, they parse saved result tables and audits.

## Implementation contract

The current renderer lives at:

```bash
scripts/render_eeg_v1_ridge_visuals.py
```

It scans evidence bundles under `/Users/aayu/Downloads/versions` and writes:

- an artifact inventory figure;
- EEG→EEG task metric strip plots from `task_results.csv`;
- baseline MSE/rank plots from `baseline_ranking.csv`;
- leakage/eval/paper-mode gate audit plots from JSON reports;
- metrics/provenance JSON;
- Markdown analysis page.

Run it with:

```bash
PYTHONPATH=src python3 scripts/render_eeg_v1_ridge_visuals.py \
  --versions-root /Users/aayu/Downloads/versions \
  --out-dir docs/research/eeg_v1_ridge_visuals
```

## Caveats

```{warning}
The current visual packet is an evidence-artifact audit. It is useful for showing what the saved runs report and what validations passed. It is not a raw EEG physiology figure, not clinical evidence, and not a seizure prediction claim.
```

## What the result should teach

If ridge ranks well in saved baseline tables, the next scientific step is not to claim a stronger brain model. The next step is to save the tensors and predictions needed for harder diagnostics: held-out subject/site/dataset splits, longer-horizon non-overlap controls, permutation nulls, residual PSDs, and real per-window overlays with provenance.
