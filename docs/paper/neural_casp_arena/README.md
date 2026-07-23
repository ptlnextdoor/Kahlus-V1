# Neural-CASP arena paper packet

**Thesis:** Residual forecastability claims in noninvasive neural EEG/PSG do not
reproduce under leakage-controlled, subject-held-out evaluation. The arena that
produces honest positives *and* honest negatives is the product.

Kahlus is not a clinical predictor, consciousness detector, or foundation model.
It is a **lie detector for neural forecasting claims** — Neural-CASP gates,
residual forecastability score (RFS bits), copy-trap / overlap audit, and
artifact contracts (`claim_scope`, `stop_reason`).

## What this packet contains

| File | Purpose |
| --- | --- |
| [findings.md](findings.md) | F0–F5 findings table with tags, numbers, claim scopes |
| [methods_gate.md](methods_gate.md) | Gate predicate, baseline ladder, control contract |
| [../ieee_figure_source/](../ieee_figure_source/) | Figure source packet (regenerate from committed CSVs) |

Machine-readable ledger: [`docs/results/findings-ledger.md`](../../results/findings-ledger.md).

## Headline numbers (honest)

| Finding | Verdict | Headline |
| --- | --- | --- |
| F0 Arena (M0–M5) | Works | Gates pass/fail honestly on synthetic + smoke |
| F1 Overlap illusion | Invalidated | 3.116 MSE / 0.972 r not valid forecast skill |
| F2 Isolated forecast | Negative | GRU loses to persistence/ridge (Sleep-EDF + BNCI) |
| F3 Interoception scout | Negative | Peripheral channels add no residual on Sleep-EDF |
| F4 Passive PCI (sleep) | Powered negative | Complexity hurts vs spectral (78 subj, n_boot=2000) |
| F5 Propofol PCI | TBD | OpenNeuro ds005620 awake vs sedated |

## Figures

Render from IEEE figure source:

```bash
python3 docs/paper/ieee_figure_source/render_all.py
```

| Figure | Shows |
| --- | --- |
| `fig1_core_task` | RFS horizon sweep — known signal vs null |
| `fig2_nfc_schematic` | Neural-CASP gate ladder |
| `fig3_gate_protocol` | Passive PIC instrument worlds |
| `fig4_mse_bar` | Product thesis + claim scope |

## Supported claim

> Leakage-aware benchmark harness for neural translation, EEG forecasting,
> state-transition forecastability, and evidence-gated reporting under public-data
> smoke — including honest nulls.

## Never claim

Clinical seizure prediction · consciousness solved · PCI/TMS replacement ·
3.116 MSE as whole-model performance · Sleep-EDF scout as Coleman replication.

## Git tags

Each finding is pinned: `finding/neural-casp-gate-suite-v1` through
`finding/passive-pci-negative-v1` (and `finding/propofol-pci-*-v1` when F5 lands).
See findings.md for commit SHAs.
