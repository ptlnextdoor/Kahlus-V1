# EEG/ridge versions evidence figures

## Real evidence artifacts

Generated from `/Users/aayu/Downloads/versions` by `scripts/render_eeg_v1_ridge_visuals.py --versions-root ...`.

- Evidence bundles scanned: **29**
- Task-result rows: **15**
- EEG→EEG task rows: **12**
- Baseline-ranking rows: **80**
- Audit rows: **33**
- Raw tensor or prediction arrays found: **No**

No raw tensor or prediction-array artifact was found, so this renderer intentionally does **not** generate a prediction overlay, waveform trace, or synthetic EEG window figure.

## EEG task metrics

- `future_state_forecasting`: n=6, median Pearson r=0.664, median $R^2$=0.478, median test MSE=28
- `masked_neural_reconstruction`: n=6, median Pearson r=0.415, median $R^2$=0.155, median test MSE=45.3

## Baseline rankings

- `linear_ridge`: n=8, median MSE=7.78, median rank=1
- `autoregressive_ridge`: n=8, median MSE=18.4, median rank=2.5
- `mlp`: n=8, median MSE=26.1, median rank=3.5
- `tcn`: n=8, median MSE=29.7, median rank=4.5
- `persistence`: n=8, median MSE=30.9, median rank=5.5
- `transformer`: n=8, median MSE=42.4, median rank=5.5
- `neurotwin`: n=8, median MSE=45, median rank=6.5
- `ssm_fallback`: n=8, median MSE=46.7, median rank=7.5
- `train_mean`: n=8, median MSE=53.6, median rank=8.5
- `random_permutation`: n=8, median MSE=84.6, median rank=10

## Audit summary

- Total violations recorded in parsed audits: **0**
- Total warnings recorded in parsed audits: **0**

## Figure files

- `fig01_versions_evidence_inventory.png/.pdf`
- `fig02_eeg_task_metrics_from_versions.png/.pdf`
- `fig03_real_baseline_ranking.png/.pdf`
- `fig04_leakage_and_gate_audit.png/.pdf`
