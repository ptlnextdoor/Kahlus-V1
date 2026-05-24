# Experiments

Planned experiments:

- Future-state forecasting.
- Masked neural reconstruction.
- Cross-modal translation where paired modalities exist.
- Few-shot subject adaptation.
- Dataset/site generalization.

Baselines include ridge, MLP, TCN, Transformer, SSM fallback, modality specialists, and clearly labeled competitor references.

Current executable prepared-manifest suite:

- ranks ridge, MLP, TCN, Transformer, SSM fallback, and NeuroTwin on forecasting, reconstruction, and cross-modal translation;
- includes bootstrap confidence intervals for MSE/MAE in supervised prepared-task artifacts;
- reports auxiliary few-shot subject adaptation and dataset/site generalization metrics;
- labels synthetic prepared outputs as plumbing, not scientific evidence;
- requires `nt eval audit` over event and split manifests before interpreting prepared benchmark artifacts.
