# Rigor rubric: Coleman EGG–sleep paper ↔ Neural-CASP

Reference: Rao et al. 2025, *Simultaneous stomach-brain electrophysiology reveals dynamic coupling in human sleep* (bioRxiv 2025.11.13.686572). Stanford Coleman lab. Used as a **rigor bar**, not a replication target.

## What the reference paper does well

| Technique | Coleman usage | Kahlus equivalent |
| --- | --- | --- |
| Covariate adjustment | LMM with sex, condition, time-since-meal, repeated measures | DML nuisance block **B** + cross-fitting (`m1._crossfit_residual_proba`) — **out-of-fold** |
| Multiple comparisons | Cluster-based permutation (p < 0.05 corrected) | `m4._cluster_permutation_rfs` on patient clusters |
| Surrogate nulls | PAC compared to phase-shuffled null | Circular-shift / time-shift / label-shuffle controls |
| Effect sizes | Cohen's d, fold-change, von Mises κ | RFS bits + cluster-bootstrap 95% CI (`n_boot≥2000` claim mode) |
| Hierarchical structure | Night nested in subject | Subject as cluster unit; GroupKFold cross-fit |
| Predictive claim | EGG ISO variance → subjective sleep quality beyond PSG+cardiac (ΔR²=0.13, n=49 nights) | **Reformulated:** subject-held-out RFS of autonomic block **Z** for arousal/sleep-quality **Y** beyond cortical spectral **B** |

## Where Kahlus exceeds the reference (the paper's intellectual core)

| Axis | Coleman | Neural-CASP |
| --- | --- | --- |
| Holdout | In-sample partial correlation / ΔR² | Subject-held-out cross-fitted RFS |
| Generalization | Single cohort (60 participants, 3 nights) | Dataset-held-out (MESA → SHHS) |
| Baseline ladder | PSG + cardiac covariates | Must beat **best** trivial (moving average, persistence, ridge, nuisance-only) |
| Negative controls | Surrogate for PAC only | Shuffle + time-shift + circular-shift + subject-probe + synthetic null |
| Leakage audit | None on predictive claim | Overlap/copy-trap (F1); M0 Amrith 127→128 acceptance |
| Claim hygiene | Broad interoception narrative | Scoped `claim_scope` + `stop_reason` in every artifact |

## Reformulated estimand (flagship F6)

```text
Y = micro-arousal event within horizon h epochs (binary)
    OR night-level arousal index / sleep-quality proxy (continuous → binned)

B = cortical spectral baseline (bandpower, entropy, 1/f slope) + cycle/history nuisance
Z = autonomic/peripheral block (HRV from ECG, respiration, EOG, EMG infraslow features)

Question: RFS_bits = I(Y; Z | B, S) > 0 on held-out subjects?
Secondary: train MESA, test SHHS (dataset-held-out)
```

Sleep-stage discrimination (F4) and stage-transition scout (F3) are **honest negatives** — the target was too easy or wrong shape. Arousal/outcome-linked Y is the reformulation.

## Gate pass predicate (unchanged)

```text
GatePass = split_disjoint ∧ finite ∧ beat_best_baseline ∧ controls_pass ∧ powered ∧ scoped_claim
```

Power bar: ≥8 held-out subjects, ≥100 positive windows, `bootstrap_mode=claim`, controls collapse to <40% of headline RFS.

## Coleman collab stretch (not flagship)

Real simultaneous EGG+EEG re-analysis under held-out RFS would be highest impact but requires non-public data. Public flagship = NSRR MESA (+ SHHS held-out).
