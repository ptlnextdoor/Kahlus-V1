# NeurIPS Datasets & Benchmarks — submission draft

**Title:** Neural-CASP: A Leakage-Controlled Arena for Residual Forecastability in Noninvasive Brain–Body Signals

## Abstract

Noninvasive neural forecasting papers routinely report strong in-sample performance that collapses under subject-held-out evaluation and trivial baselines. We introduce **Neural-CASP**, a blind benchmark arena with a cross-fitted residual forecastability score (RFS bits), a baseline ladder that must beat moving-average persistence, mandatory negative controls, and an overlap/copy-trap audit that retires inflated next-sample metrics. We demonstrate the arena on six findings (F0–F6): synthetic validation, an overlap illusion that invalidated a 0.972 Pearson headline, powered honest negatives on sleep-state complexity and interoception scouts, and a reformulated autonomic arousal defendant on NSRR polysomnography. Under strict held-out evaluation, residual forecastability claims that dominate the literature do not reproduce; the arena that produces honest positives and honest negatives is the contribution.

## Contributions

1. **Neural-CASP benchmark artifact** — RFS estimator, gate predicate, control contract, `runner → gate → JSON + Markdown` evidence pattern.
2. **Overlap illusion audit (F1)** — demonstrates 126/127 input-target overlap inflates forecast skill; constitutionalized Amrith 127→128 acceptance in M0.
3. **Arena bites (F2–F5)** — isolated forecast negative, interoception scout negative, Passive PCI powered negative, propofol cross-etiology probe.
4. **Reformulated autonomic defendant (F6)** — micro-arousal RFS beyond cortical spectral baseline on NSRR MESA (+ SHHS dataset-held-out); positive or powered negative.

## Figures

| Fig | Source | Content |
| --- | --- | --- |
| 1 | `ieee_figure_source/fig1` | RFS horizon sweep known vs null |
| 2 | `ieee_figure_source/fig2` | Neural-CASP gate ladder |
| 3 | `ieee_figure_source/fig3` | Passive PIC instrument worlds |
| 4 | `ieee_figure_source/fig4` | Product thesis + claim scope |
| 5 | `amrith_overlap_contrast` | OLD overlapping vs NEW isolated forecast MSE |

## Claim boundaries

Supported: leakage-aware benchmark harness; honest negatives; scoped residual forecastability under public-data smoke.

Never: clinical diagnosis · consciousness solved · TMS-PCI replacement · 3.116 MSE as whole-model · Coleman EGG replication without data.

## Artifacts

- Findings ledger: `docs/results/findings-ledger.md`
- Arena packet: `docs/paper/neural_casp_arena/`
- Rigor rubric: `docs/paper/neurips_2026/rigor_rubric.md`
- Git tags: `finding/*-v1`

## Test plan

```bash
PYTHONPATH=src python -m pytest -q tests/forecastability/test_autonomic_rfs.py tests/adapters/test_nsrr.py tests/forecastability/test_m0.py
uv run ruff check .
```
