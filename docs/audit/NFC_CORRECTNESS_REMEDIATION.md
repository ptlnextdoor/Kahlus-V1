# NFC Correctness Remediation

## Scope

This record covers software-correctness findings in the experimental NFC, pair-operator, local sequence baselines, and prepared probabilistic-training path. It does not authorize a Neural Field Compiler, model-superiority, calibration, biological-state, or clinical claim.

## Finding Disposition

| Finding | Repair | Regression evidence | Scientific status |
|---|---|---|---|
| NFC `backbone` and `n_layers` were inert | `FieldUpdateOperator` now dispatches to an actual causal GRU or causally masked Transformer and receives layer/head configuration | backend/depth test compares computation and layer count | implemented, not empirically validated |
| Symmetric fMRI convolution read future samples | shared `CausalHRFAdapter` uses explicit delay and left-only temporal support | future-perturbation tests for NFC and pair-operator | causality contract tested locally |
| Constant `expert_utilization` looked like measured routing | field removed; NFC reports learned modality weights and the pair path reports its exact low-rank mixing matrix when materialized | output-contract tests reject the old field | fixed |
| NFC chose one alphabetically first source | all available configured sources are projected, validated, and fused by learned softmax weights | perturbing either source changes output | implemented, not a missing-modality result |
| Task and subject-ID inputs were discarded | task embedding is active; subject IDs fail closed in NFC | task/subject boundary tests | fixed for NFC |
| Geometry could be ignored | configured coordinates are required, shape/finite checked, and encoded; malformed structural priors fail | coordinate and structural-prior tests | coordinate conditioning implemented; field claim unvalidated |
| Uncertainty head was not scored | prepared training uses Gaussian negative log-likelihood when NFC/pair uncertainty is enabled | uncertainty-head gradient and prepared-loop tests | probabilistic training implemented; calibration unvalidated |
| Tiny Transformer and TCN were noncausal | shared causal mask/positions and left-only convolutions | future-perturbation tests | fixed |
| `TinySSM` was a GRU alias | TinySSM is now a stable diagonal recurrent state-space model; historical `ssm_fallback` is explicitly a GRU | type and causality tests | distinct local baselines |
| `model` duplicated `neurotwin` | duplicate executable row removed | runner-ID uniqueness test | fixed |
| Bare `ssm` silently meant GRU | ambiguous name now raises; callers choose `tiny_ssm`, `gru`, or `ssm_fallback` explicitly | construction tests | fixed |

## Remaining Scientific Blocks

1. The NFC is a coordinate-conditioned finite sensor model, not a demonstrated neural operator.
2. No montage-transfer or discretization-invariance experiment has run.
3. Gaussian likelihood training does not establish calibration; validation-fitted calibration and external coverage are required.
4. Synthetic suites test plumbing and falsification behavior only.
5. The HNPH baseline feasibility gate must pass before any NFC or A100 model experiment is authorized.
6. The primary HNPH contribution remains the externally evaluated conditional-information question, not this architecture.

## Claim Boundary

Permitted implementation wording:

> The repository contains an experimental coordinate-conditioned latent-tensor model with causal temporal backends, same-time sensor attention, modality-specific readouts, and likelihood-trained predictive scale.

Blocked wording includes neural operator, recovered biological field, calibrated uncertainty, state-of-the-art model, clinical forecaster, or validated Neural Field Compiler.
