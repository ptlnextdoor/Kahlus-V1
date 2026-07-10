# Experiment Validity Audit

## Recovered EEG Result

The read-only A100 evidence release reports a BNCI2014-001 subject-separated run with nine subjects and 2,592 events. Its selected 100,000-step model reports future-task MSE 3.1161 and Pearson correlation 0.9721, compared with ridge MSE 7.745 and persistence MSE 8.204. Masked reconstruction reports MSE 53.977 and correlation approximately -0.013.

These numbers are internally recorded but not paper-ready because:

1. The forecasting target overlaps the context by 126/127 samples.
2. The model is a GRU-based translator, not NFC.
3. Baseline budgets and model-selection procedures are not matched.
4. Confidence intervals do not use subjects as the independent sampling unit.
5. The release lacks raw-file identities and training checkpoints needed for full reproduction.
6. Masked reconstruction fails, and the artifact's scientific claim gate is false.

## Forecastability Ladder

| Milestone | Data | Result | Scientific interpretation |
|---|---|---|---|
| M0 | synthetic | baseline/gate pass | instrumentation only |
| M1 | synthetic known/null | pass with numerical warnings in residual offset path | falsification instrumentation only |
| M2 | tiny Sleep-EDF subset | six pairs; gate pass | machinery check, underpowered |
| M3 | tiny CHB-MIT subset | underpowered; gate false; site nuisance accuracy 1.0 | negative feasibility result |
| M4 | synthetic | synthetic pass, real run absent | synthetic only |
| M5 | synthetic | pass without external data | synthetic only |

## Statistical Defects

- Window and element counts are used where subject/recording clusters are the scientific units.
- No prespecified subject-level primary endpoint is enforced.
- No paired subject-level bootstrap or hierarchical model supports the main model-versus-baseline claim.
- Seed and dataset variance are not characterized for the recovered model.
- Hyperparameter search and model selection are not separated from final evaluation.
- No multiplicity plan exists for the expanding task/model/dataset matrix.

## Required Replacement Protocol

1. Freeze a raw-file manifest with hashes and acquisition metadata.
2. Split by subject/patient before all fitted preprocessing and before window construction.
3. Define context `[t-L,t]` and target `[t+g,t+g+H]` with `g>0`; assert disjoint sample ranges.
4. Fit normalization, channel imputation, and feature selection on training subjects only.
5. Tune all models under a common validation budget; preserve failed runs.
6. Evaluate persistence, seasonal persistence where meaningful, ridge/AR/VAR, TCN, GRU, Transformer, TinySSM only if it is a real SSM, shuffled target, time shift, identity, and artifact controls.
7. Report per-subject metrics and paired subject-level confidence intervals.
8. Repeat across at least two datasets with an external-dataset test.
9. Use a locked environment and one-command raw-to-report reproduction.

## Clinical Validity

No current experiment supports seizure prediction, sleep diagnosis, depression/anhedonia assessment, treatment response, or clinical decision support. Clinical utility would require prospectively defined endpoints, representative cohorts, calibration, prevalence-aware metrics, and external prospective validation.
