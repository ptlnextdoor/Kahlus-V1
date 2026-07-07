# M4 Sleep-EDF Primary-Horizon Pre-Analysis Plan

This note documents the committed M4 Sleep-EDF execution contract. It is a prospective
pre-analysis plan for a future local/public-data run, not a completed Sleep-EDF result.

## Scope

- Dataset: PhysioNet Sleep-EDF Expanded.
- Raw data policy: raw PSG and hypnogram files must stay outside the repository.
- Split/cluster unit: dataset-scoped Sleep-EDF subject ID parsed from the filename.
- Session boundary unit: subject/night ID parsed from the filename.
- Primary horizon: horizon 1.
- Descriptive horizons: horizons 2 and 3.
- Primary endpoint: RFS bits versus the strongest gated nuisance/trivial baseline.
- Claim boundary: benchmark-method hardening only. This does not permit clinical,
  diagnostic, treatment, or model-superiority claims.

## Required Gates

The primary horizon is the only inferential horizon in this protocol. Other horizons
are descriptive unless a separate multiplicity-controlled plan is added.

The future Sleep-EDF execution must fail closed if:

- the local Sleep-EDF root is inside the repository,
- Sleep-EDF filenames cannot be parsed into subject and night metadata,
- primary-horizon event patients are below 8,
- primary-horizon positive events are below 100,
- the primary RFS confidence interval includes zero,
- shuffled-target or time-shift controls are too close to primary RFS,
- the patient-cluster sign-flip permutation is not significant,
- any nuisance probe exceeds chance plus 0.20.

## Provenance Requirements

The runner writes redacted execution metadata only. It records file names, file
hashes, parsed subject IDs, parsed night/session IDs, pair counts, the
preregistration hash, the primary-horizon result, and gate failures. It does not
record absolute local raw-data paths.

## Sources

- PhysioNet Sleep-EDF Expanded v1.0.0 documents the public PSG/hypnogram dataset,
  EDF format, and manual sleep-stage scoring.
- EEG leakage literature motivates subject/session-aware splits and fail-closed
  nuisance probes because segment-level random splits can overestimate performance.
- Human medical intervention evidence, such as randomized-trial meta-analysis of
  exercise for depression, is relevant only to treatment claims. This M4 feature
  makes no intervention, depression, clinical, or treatment claim.
