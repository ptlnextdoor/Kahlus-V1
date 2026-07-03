# Overnight Proposals - 2026-07-03

## Recon Sources

- Consensus: EEG foundation model benchmarking remains unsettled; larger models do not reliably improve cross-subject generalization, and specialist baselines remain competitive: [EEG Foundation Models: Progresses, Benchmarking, and Open Problems](https://consensus.app/papers/eeg-foundation-models-progresses-benchmarking-and-open-liu-chen/a582970c22555ee282d3c4efb2d9f33d/?utm_source=chatgpt).
- Consensus: segment/window leakage can strongly inflate translational EEG results: [Data leakage in deep learning studies of translational EEG](https://consensus.app/papers/data-leakage-in-deep-learning-studies-of-translational-eeg-brookshire-kasper/7037ef49c9b752acb7b5d3a2a4c7ce31/?utm_source=chatgpt).
- Consensus: log-loss forecastability across horizons is an information-theoretic object, not just a model leaderboard: [Forecastability as an Information-Theoretic Limit on Prediction](https://consensus.app/papers/forecastability-as-an-informationtheoretic-limit-on-catt/69177125e0f150c3b4de603c2fb217bd/?utm_source=chatgpt).
- Web: official data sources are CHB-MIT and Sleep-EDF on PhysioNet.
- opensrc: checked `scikit-learn/scikit-learn` logistic source and `scipy/scipy` optimizer source. Decision: do not add SciPy because this repo does not depend on it; use a small IRLS offset-GLM.

## Candidate Scores

| candidate | novelty | feasible tonight | impact if true | leakage risk | decision |
|---|---:|---:|---:|---:|---|
| Leakage-safe forecastability-vs-horizon curve | 4 | 5 | 4 | 2 | chosen |
| Nuisance-invariant residual representation with probe gate | 4 | 3 | 4 | 3 | defer |
| Cross-modal missing-modality recoverability/identifiability gate | 5 | 2 | 5 | 3 | defer |

## Chosen Candidate

Build `M4`: a forecastability-vs-horizon information curve. Labels are shifted within each patient only, then RFS is recomputed per horizon against the best nuisance/trivial baseline. This directly extends the existing RFS ladder and gives the repo a compact benchmark-method contribution without adding a new model family.

## Why The Other Two Wait

Nuisance-invariant representation learning is useful, but the current M3 artifact already shows the cheaper issue: the model must first beat moving-average/history baselines. Cross-modal recoverability is closer to the broad NeuroTwin v1 claim, but needs paired multimodal real data and stronger baseline coverage than this workspace can honestly create overnight.
