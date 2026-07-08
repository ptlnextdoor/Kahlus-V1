# Limitations and claim boundaries

<span class="limit-badge">READ BEFORE CLAIMS</span>

Kahlus is a research codebase. It is not a medical device, a clinical diagnostic tool, or a validated brain foundation model.

## Current safe claims

- The repo contains reproducible benchmark plumbing for neural translation tasks.
- The EEG v1 ridge visualizations explain an existing future-window benchmark and its autocorrelation risks.
- Synthetic fixtures are useful for testing code paths, not for scientific claims.
- Leakage-safe split manifests and audits are first-class design goals.

## Current unsafe claims

Do not claim that Kahlus:

- understands brain state;
- predicts seizures clinically;
- diagnoses depression or any disease;
- is a digital twin of a patient;
- beats expert clinical systems;
- proves a new neural field model from synthetic or short-horizon EEG baselines.

## Expert caveat

The current EEG v1 ridge result is best read as a sanity check for benchmark geometry. Short forecast horizons, overlapping windows, normalization scale, and smooth synthetic dynamics can make persistence or ridge baselines look strong.
