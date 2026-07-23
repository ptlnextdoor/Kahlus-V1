# Neural-CASP gate methods (pointer)

Full implementation lives in `src/neurotwin/forecastability/`. This document
states the contract; do not duplicate logic here.

## Residual forecastability score

```text
RFS_bits = (NLL_B - NLL_{B+Z}) / ln(2)  ≈  I(Y; Z | B)
```

- Cross-fitted nuisance model on **B**, residual offset model on **Z**
- Subject-cluster bootstrap CI (`n_boot=2000` in claim mode)
- Cluster-permutation null on RFS (`m4._cluster_permutation_rfs`)

Canonical: [`m1.py`](../../../src/neurotwin/forecastability/m1.py) —
`_crossfit_residual_proba`, `_rfs_payload`, `_cluster_bootstrap_rfs`.

## Baseline ladder (must beat the best)

```text
moving_average → persistence → base_rate → nuisance_only (logistic on B)
```

Best baseline selected by lowest NLL on held-out folds. Never claim residual
signal without beating this ladder.

## Control contract (must collapse)

| Control | Mechanism | Pass criterion |
| --- | --- | --- |
| Label shuffle | Shuffle Y within cross-fit train folds | RFS < 40% of headline |
| Time shift | Shift train labels by 9 epochs | RFS < 40% of headline |
| Circular-shift surrogate | Roll Z within subject | RFS < 40% of headline |
| Subject probe | Nearest-centroid on Z | accuracy ≤ chance + 0.2 |
| Synthetic null | Known-null fixture | \|RFS\| < 0.03, CI high ≤ 0.05 |

## Passive PCI state discrimination

Task: **Y = current macrostate** (one-vs-rest), not transition forecasting.

| Block | Features | Module |
| --- | --- | --- |
| Spectral baseline B | bandpower, spectral entropy, line length, 1/f slope | `m1.handcrafted_eeg_features` + `complexity_features.spectral_slope_block` |
| Complexity block Z | LZ, permutation entropy, multiscale entropy | `complexity_features.complexity_block` |

Sleep substrate: [`passive_pci.py`](../../../src/neurotwin/forecastability/passive_pci.py)  
Propofol substrate: [`propofol_pci.py`](../../../src/neurotwin/forecastability/propofol_pci.py)

## Power bar (constitution)

- ≥8 held-out subjects (cluster units)
- ≥100 positive windows per powered state
- `bootstrap_mode=claim` → `n_boot ≥ 2000`
- `claim_scope` + `stop_reason` in every JSON artifact

## Runner pattern

```text
runner → gate → JSON + Markdown
```

Example:

```bash
PYTHONPATH=src python scripts/run_passive_pci_gate.py \
  --sleep-edf-root /path/to/sleep-cassette \
  --bootstrap-mode claim
```
