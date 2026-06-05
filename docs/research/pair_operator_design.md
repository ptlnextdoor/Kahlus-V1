# NeuroTwin Pair-Operator Design

Status: local hardening complete enough for 1x A100 debug rehearsal
Scope: Algonauts/CNeuroMod fMRI-first architecture experiment
Claim scope: no model superiority claim until real A100 evidence exists

## Problem Statement

Naturalistic brain activity is not a flat token stream. It is a time-evolving
relational system over parcels, sensors, or regions, driven by stimulus streams
and constrained by evidence quality. The Pair-Operator experiment tests whether
explicit low-rank parcel-pair state improves fMRI tasks under leakage-audited
splits.

The north-star benchmark is Algonauts/CNeuroMod fMRI first, with three tasks:

1. `future_state_forecasting`
2. `masked_neural_reconstruction`
3. `stimulus_to_fmri_response`

The allowed future paper claim is narrow:

> Pair-Operator improves at least one core fMRI/stimulus task under leakage-audited
> splits versus strong local baselines.

No current local evidence supports model superiority, SOTA, clinical use, first
brain foundation model, first multimodal brain model, or first stimulus-to-brain
claims.

## Prior Art Table

| System | Relevant idea | Constraint on NeuroTwin claims |
| --- | --- | --- |
| TRIBE v2 | Tri-modal video/audio/language modeling for fMRI prediction. | Stimulus-to-fMRI is crowded; NeuroTwin must not claim first stimulus-to-brain. Use `tribe_style_clean_room` only as a local approximation unless exact upstream reproduction is added. |
| BrainVista | Naturalistic fMRI next-token prediction with history/stimulus masking and long-horizon rollout framing. | BrainVista-style rollout is a baseline lane, not empty territory. NeuroTwin uses `brainvista_style` as a causal local approximation, not exact BrainVista. |
| Brain-OF | fMRI/EEG/MEG foundation-style model with any-resolution sampling and sparse MoE. | NeuroTwin must not claim first multimodal brain model or uncontested foundation-model novelty. |
| BrainOmni | EEG/MEG tokenizer and sensor-aware modeling. | NeuroTwin must not claim first sensor-aware neural model. |
| DeepSeek-V3 | Efficient conditional computation via sparse MoE and latent attention. | Efficiency and conditional computation matter; Pair-Operator should expose low-rank and block controls instead of dense all-pair memory. |
| AlphaFold-style Pairformer idea | Separate node and pair states plus confidence-style outputs can be powerful in relational domains. | Pair-Operator borrows the relational-state lesson, not protein-domain claims or exact Pairformer mechanics. |

## What Pair-Operator Adds

NeuroTwin Pair-Operator adds an executable, leakage-gated fMRI architecture lane
with:

- node states `z_i(t)` for parcel activity/state
- low-rank pair states `P_ij(t)` represented by learned left/right factors
- a pair update path that actively changes node updates and predictions
- causal stimulus alignment for stimulus-to-fMRI tasks
- per-region/per-time uncertainty outputs and optional pair uncertainty
- explicit evidence gates for leakage, stimulus provenance, non-finite failures,
  baseline ranking, and final claim eligibility

This is not a generic model zoo. The experiment is useful only if the pair path
changes outcomes under real, audited fMRI splits.

## Architecture Diagram

```text
Prepared neural windows
  ├─ fMRI parcel windows: x_fmri[t, i]
  ├─ optional stimulus features: s[t, k]
  └─ metadata: subject/session/site/dataset/split/stimulus provenance

Node encoder
  x_source[t, *] -> projected parcel values -> node states z_i(t)

Stimulus alignment adapter
  causal lag/HRF-style stimulus drive -> additive node drive

Low-rank pair state
  left factors  L_i,r
  right factors R_j,r
  optional top-k sparsity over R
  optional network block gates over L
  P_ij is implicit: softmax((L_i · R_j) / sqrt(rank))

Pair-to-node operator
  pair_context_i(t) = sum_r L_i,r * sum_j R_j,r * z_j(t)
  z_i(t) <- z_i(t) + MLP(pair_context_i(t))

Multi-timescale temporal operator
  per-parcel GRU/SSM/Transformer path over time
  fMRI HRF context path

Task heads
  future_state_forecasting
  masked_neural_reconstruction
  stimulus_to_fmri_response

Uncertainty and evidence
  per-region/time uncertainty
  optional pair uncertainty
  evidence_gate.json only after explicit finalization
```

## State Definitions

Node state:

```text
z_i(t) ∈ R^d
```

`i` indexes an fMRI parcel or sensor. `t` indexes the prepared window time step.
The implementation builds node states by projecting the source modality onto the
target parcel axis, encoding scalar parcel values, adding optional causal
stimulus drive, and then applying pair and temporal updates.

Pair state:

```text
P_ij(t) ≈ f(L_i, R_j, z_j(t))
```

The implementation does not store a dense learned `N x N` parameter table. It
stores low-rank factors:

```text
L ∈ R^(N x r)
R ∈ R^(N x r)
```

The active pair context is computed as:

```text
summary_r(t) = sum_j R_j,r z_j(t)
context_i(t) = sum_r L_i,r summary_r(t)
```

This keeps the node update path `O(N * r * d)` instead of `O(N^2 * d)`. Dense
pair confidence is materialized only for small parcel counts. For 1000 fMRI
parcels, `pair_confidence` is compact with shape `(2, 1000, rank)`.

## Low-Rank Controls

Required controls are now explicit in config:

- `use_pair_state`: disables the pair-to-node update path
- `pair_rank`: low-rank factor width
- `pair_top_k`: optional rank-wise top-k sparsity on right factors
- `network_blocks`: optional network/block gates over left factors
- `pair_confidence_max_parcels`: maximum dense confidence map size
- `use_pair_uncertainty`: optional pair uncertainty output

The no-pair ablation keeps the same node/temporal path and disables the active
pair update. Runtime estimates mark it as `disabled_low_rank_parameters_present`
because the current module still allocates pair parameters for config symmetry.

## Multi-Timescale Operator

The current temporal path is intentionally small:

- per-parcel recurrent/SSM fallback or Transformer path over window time
- fMRI HRF context adapter
- refinement head for prediction updates

The A100 run should not tune performance locally. The only local requirement is
that the path is shape-safe, finite, and ablation-ready.

## Stimulus Alignment Adapter

Stimulus-to-fMRI is evidence-gated. The local BrainVista-style approximation is:

```text
history fMRI + optional lagged stimulus features -> future fMRI
```

Default stimulus features are history-only:

- `stimulus_lag_steps=1`
- `include_current_stimulus=false`
- `hrf_lag_steps=2`

Tests verify that future and current stimulus values do not affect the current
BrainVista-style feature row by default.

## Uncertainty Head

Pair-Operator emits:

- `uncertainty`: per-time/per-region uncertainty with prediction shape
- `pair_uncertainty`: optional compact or dense pair uncertainty
- `uncertainty_calibration.csv`: task-level mean uncertainty and
  error/uncertainty correlation when metrics are finite

Uncertainty metrics are diagnostic. They do not make a run claim-eligible by
themselves.

## Evidence Gate

Stimulus claim eligibility is impossible through transcript hashes, synthetic
embeddings, or self-attested embedding hashes. For `claim_eligible=true`, every
used stimulus feature window must provide:

- `stimulus_feature_source`
- `stimulus_feature_modalities`
- `stimulus_feature_hash`
- `stimulus_feature_status` in a real/precomputed class
- local source artifact path, manifest, or `file://` URI
- computed source artifact hash equal to declared hash

If any required item is missing, the task is plumbing-only and
`claim_eligible=false` with exact failure reasons.

Training writes provisional evidence only:

- `evidence_gate_provisional.json`
- `diagnostic_report_provisional.md`

Final evidence is written only by explicit finalization:

```bash
PYTHONPATH=src python3 -m neurotwin.cli report evidence-gate --run-dir /abs/run
```

`report` and `report model-card` are read-only and must not mutate
`summary.json`. Baseline ranking validity comes from structured
`prepared_baseline_suite.json`, not CSV string scanning.

## Non-Finite Safety

Training now fails closed for non-finite behavior:

- non-finite loss is detected
- non-finite gradient norm is detected
- optimizer step is skipped on non-finite loss or gradient
- default clipping is `max_grad_norm=1.0`
- task quarantine forces `scientific_claim_allowed=false`
- aggregate metrics ignore NaN values but list quarantined tasks
- best finite checkpoint rollback is recorded separately from final metrics
- final reports distinguish completed training, task quarantine, and claim
  disallowance

## Baselines

Executable local baselines:

- `train_mean`
- `persistence`
- `linear_ridge`
- `autoregressive_ridge`
- `neurotwin`
- `tribe_style_clean_room`
- `brainvista_style`
- `pair_operator`
- `pair_operator_no_pair_state`

Non-executable catalog baselines remain labeled as upstream/contextual when no
local runner exists. Do not describe `tribe_style_clean_room` as exact TRIBE v2.
Do not describe `brainvista_style` as exact BrainVista.

## Ablation Matrix

| Variant | Purpose | Expected artifact |
| --- | --- | --- |
| `current_neurotwin` | Existing translator baseline. | baseline ranking row |
| `brainvista_style` | Causal history/stimulus fMRI approximation. | baseline ranking row |
| `pair_operator_no_pair_state` | Node and temporal path without pair update. | `pair_operator_ablation.csv` |
| `pair_operator_low_rank_pair_state` | Pair update on, uncertainty/refinement off. | `pair_operator_ablation.csv` |
| `pair_operator_pair_state_stimulus_adapter` | Pair update plus stimulus adapter. | `pair_operator_ablation.csv` |
| `pair_operator_pair_state_uncertainty` | Pair update plus uncertainty diagnostics. | `uncertainty_calibration.csv` |
| `pair_operator_full` | Full candidate. | all final sidecars |

## Configs

Required configs are present:

- `configs/train/algonauts_pair_operator_debug.yaml`
- `configs/train/algonauts_pair_operator_full.yaml`
- `configs/train/algonauts_pair_operator_ablation_array.yaml`
- `configs/train/algonauts_pair_operator_no_pair.yaml`
- `configs/train/algonauts_brainvista_style_debug.yaml`
- `configs/train/algonauts_current_neurotwin_debug.yaml`

Each includes explicit dataset placeholders, `run_id`, `max_grad_norm`, `bf16`,
evidence gate settings, `require_real_stimulus`, and transcript-hash denial.

## A100 Memory Estimates

Dry-run estimates from the local config resolver:

| Config | Params | Activation MB | Optimizer MB | Checkpoint MB | Pair state MB | 1x A100 MB | 6x DDP per GPU MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| debug | 3,861,872 | 8.000 | 29.464 | 7.366 | 0.061 | 44.830 | 44.830 |
| full | 3,849,584 | 16.000 | 29.370 | 7.342 | 0.061 | 52.712 | 52.712 |
| no-pair | 3,849,584 | 8.000 | 29.370 | 7.342 | 0.061 | 44.712 | 44.712 |

These are planning estimates only, not measured A100 memory. They exclude data
loader overhead, CUDA workspaces, and framework fragmentation.

## Exact Command Helper

The print-only helper emits the 1x debug, 6x full, and one-GPU-per-ablation
commands without submitting jobs:

```bash
PREPARED_ROOT=/abs/prepared/algonauts_cneuromod \
RUN_ROOT=/abs/persistent/neurotwin_pair_operator \
CONFIG_ROOT=/abs/persistent/neurotwin_pair_operator/configs \
PHASE1_EVAL_DIR=/abs/persistent/neurotwin_pair_operator/eval/phase1_paper_mode \
bash scripts/print_pair_operator_a100_commands.sh
```

The launchers require materialized absolute config paths and persistent
absolute `RUN_ROOT`. The full 6-GPU path does not run paper-mode inside the full
allocation unless `A100_RUN_PAPER_MODE_IN_FULL=1`.

## Failure Modes

- Pair state does not beat the no-pair ablation on any audited fMRI task.
- BrainVista-style local approximation beats Pair-Operator on rollout.
- Stimulus evidence remains plumbing-only because source artifacts cannot be
  verified.
- A required task is quarantined due to non-finite loss, gradient, or metric.
- Structured baseline suite is missing or contains only unavailable rows.
- Pair confidence becomes uninterpretable or unstable under 1000 parcels.
- Uncertainty is finite but uncalibrated or anticorrelated with error.
- The full DDP run diverges although the 1x debug run passes.

## Paper Claim Criteria

An architecture-paper claim is allowed only if all are true:

1. Real prepared Algonauts/CNeuroMod manifests are used.
2. Splits are leakage-audited and subject/site/dataset policy is claim-eligible.
3. Stimulus features are source-artifact verified when stimulus claims are made.
4. Required tasks complete without quarantine.
5. Structured baseline rankings exist for the required tasks.
6. Pair-Operator beats strong baselines or its no-pair ablation on at least one
   core fMRI/stimulus task under the audited split.
7. Model card reports split policy, leakage audit, stimulus evidence, identity
   risk, uncertainty diagnostics, and final evidence gate.
8. Claims avoid first/SOTA/clinical language.

## Kill Criteria

Kill or demote the architecture paper if any are true:

- no-pair ablation matches or beats Pair-Operator across core tasks
- Pair-Operator only wins under invalid or ambiguous splits
- stimulus-to-fMRI evidence depends on transcript hashes, synthetic embeddings,
  or self-attested embedding hashes
- any required task is quarantined in the final run
- final evidence gate fails
- baseline ranking is unavailable or CSV-only
- pair state causes unacceptable memory/runtime overhead at 1000 parcels
- the best story is reproducibility/leakage hygiene rather than architecture

If killed, fall back to the Track A reproducibility paper: leakage demos,
model-card reporting, identity risk, and executable claim gates.
