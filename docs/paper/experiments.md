# Experiments

Planned experiments:

- Future-state forecasting.
- Masked neural reconstruction.
- Cross-modal translation where paired modalities exist.
- Few-shot subject adaptation.
- Dataset/site generalization.

Baselines include ridge, MLP, TCN, Transformer, SSM fallback, modality specialists, and clearly labeled competitor references.

NFC experiments add synthetic latent-field recovery before real-data scaling. The comparison table includes current NeuroTwin, Pair-Operator, no-pair NFC, no-observation-operator NFC, and full NFC.

Current executable prepared-manifest suite:

- ranks ridge, MLP, TCN, Transformer, SSM fallback, and NeuroTwin on forecasting, reconstruction, and cross-modal translation;
- includes bootstrap confidence intervals for MSE/MAE in supervised prepared-task artifacts;
- reports auxiliary few-shot subject adaptation and dataset/site generalization metrics;
- labels synthetic prepared outputs as plumbing, not scientific evidence;
- requires `nt eval audit` over event and split manifests before interpreting prepared benchmark artifacts.

First locked real-data protocol:

- MOABB `BNCI2014_001` with `LeftRightImagery`;
- subject-held-out split built before windowing;
- `scripts/prepare_moabb_benchmark.sh` prepares local manifests and runs audit/eval;
- A100 training uses prepared manifests only and does not download data.

Leakage-controlled EEG forecasting protocol:

- predict future EEG windows from prior windows only after the recording-level split manifest has been fixed;
- run NFC, NeuroTwin, and baseline models against the same train/validation/test manifest;
- select checkpoints on validation metrics and report test metrics only once for the final held-out split;
- compare against persistence, moving-average, nuisance-conditioned, ridge/logistic, and time-surrogate controls before interpreting any model gain;
- run shuffled-target and time-shift null controls where the evidence gate reports residual forecastability or predictive integration.

Forecastability gate boundaries:

- M4 horizon curves shift labels within patient only; labels must not cross patient boundaries when evaluating longer horizons.
- M5 Passive PIC is a synthetic instrument-validity gate. It is nuisance-conditioned and useful for testing the measurement procedure, but it is not public-data validation and does not establish external generalization.
- Any real EEG claim must point to a passing leakage audit, the exact split manifest, the baseline table, null-control results, and the final held-out report.

BIDS/OpenNeuro v1 validation is derivative-only: precomputed region/channel time-series derivatives are scanned, validated as finite 2D arrays, and converted to prepared events when present. Raw fMRI preprocessing is out of scope.
