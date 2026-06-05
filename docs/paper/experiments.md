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

BIDS/OpenNeuro v1 validation is derivative-only: precomputed region/channel time-series derivatives are scanned, validated as finite 2D arrays, and converted to prepared events when present. Raw fMRI preprocessing is out of scope.
