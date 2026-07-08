# Ridge baseline method note

```{admonition} Audience
:class: note
This page is written in two layers: plain explanation first, then equations and implementation details for technical readers.
```

## Plain explanation

Ridge regression is a linear model with a safety brake. It tries to predict the next EEG window from the current EEG window, while penalizing weights that become too large.

For the current EEG v1 future-window task, strong ridge performance does **not** automatically mean the model understands brain state. The forecast horizon is short, and the next window can overlap heavily with the current window. That means ridge may be exploiting smooth temporal continuity.

## Formal objective

Let $X \in \mathbb{R}^{n \times d}$ be the flattened input-window design matrix and $Y \in \mathbb{R}^{n \times m}$ be the flattened future-window target. Ridge solves:

```{math}
\hat{W} = \arg\min_W \lVert XW - Y \rVert_2^2 + \alpha \lVert W \rVert_2^2
```

The current visual renderer reports the exact matrix shapes used by the EEG v1 task. For the synthetic fixture currently checked into the docs, the input and target matrices are both `[1024, 48]`.

## Implementation contract

The current renderer lives at:

```bash
scripts/render_eeg_v1_ridge_visuals.py
```

It uses the existing EEG v1 future-window benchmark and writes:

- waveform input/target SVG;
- ridge feature-map SVG;
- prediction overlay SVG;
- metrics/provenance JSON;
- Markdown analysis page.

Run it with:

```bash
PYTHONPATH=src python3 scripts/render_eeg_v1_ridge_visuals.py \
  --dataset synthetic_fixture \
  --out-dir docs/research/eeg_v1_ridge_visuals
```

## Caveats

```{warning}
The current committed visual packet is synthetic fixture analysis. It is useful for explaining benchmark geometry and autocorrelation risk. It is not public EEG evidence, not clinical evidence, and not a seizure prediction claim.
```

## What the result should teach

If ridge and persistence dominate, the next scientific step is not to claim a stronger brain model. The next step is to harden the benchmark with longer horizons, non-overlap controls, held-out subject/site/dataset splits, permutation controls, and real public EEG artifacts.
