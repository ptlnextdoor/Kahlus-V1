# NeuroTwin Pair-Operator Historical Ablation Audit

Status: superseded architecture branch; retained as NFC ablation
Branch scope: Pair-Operator plus real-stimulus evidence hygiene
Claim scope: infrastructure and model experiment readiness only; no model-superiority claim

Pair-Operator is no longer the main NeuroTwin architecture. The main
experimental architecture is NeuroTwin NFC, the Neural Field Compiler. This file
is retained to document the historical Pair-Operator branch and to define the
baseline/ablation role it now plays inside NFC.

## Research Thesis

The Pair-Operator branch tested one narrow architecture bet: naturalistic brain
dynamics should be modeled as a time-evolving relational system, not only as a
flat temporal sequence. The minimal representation is:

- node state: parcel, sensor, or region activity through time
- pair state: dynamic relation between units
- stimulus drive: real precomputed stimulus features aligned to neural time
- evidence metadata: split, audit, stimulus provenance, and claim eligibility

For this ablation the implementation remains fMRI-first. EEG, calcium, spikes,
clinical labels, and additional modalities are out of scope until the fMRI path
wins or loses cleanly.

## Existing Implementation

The current branch already implements the core prototype pieces:

- `NeuroTwinPairOperator` with node tokens, low-rank pair mixing, temporal
  evolution, HRF context, refinement, pair confidence, and optional uncertainty.
- Prepared-training model selection via `model.type`, including the
  `pair_operator` alias.
- Stimulus-to-fMRI task metadata with `stimulus_evidence` attached to prepared
  tasks and surfaced into prepared-suite reports, summaries, and model cards.
- Non-finite task quarantine so NaN or Inf task metrics do not poison selected
  checkpoints, aggregate metrics, metrics JSONL, or run summaries.
- Runnable local approximation lanes for `brainvista_style` and
  `pair_operator`.
- Debug configs for Pair-Operator and real-stimulus-oriented prepared runs.

## Architecture Contract

### NeuroEventTokenizer

Current status: partially covered by `NeuralEventBatch`,
`prepared_windows_by_split`, and prepared window tasks.

Required evidence fields:

- modality
- time index or window start
- sampling rate or TR where available
- subject, session, site, and dataset identifiers
- source record and split assignment
- source/preprocessing hashes where available
- stimulus id/segment id when stimulus features are used

Gap: there is no explicit tokenizer abstraction yet. That is acceptable for the
minimal experiment, but the prepared task builder must preserve enough metadata
to audit stimulus and split claims.

### BrainPairformer

Current status: implemented as low-rank pair factors in
`NeuroTwinPairOperator`.

The current pair state avoids full `O(N^2)` parameter storage by using left and
right low-rank parcel factors and materializing a softmax pair weight matrix at
runtime. This is acceptable for the first fMRI prototype and supports the
required pair-state ablation through `use_pair_state=false`.

Known limitation: the prototype does not yet support sparse top-k graphs,
atlas/network block factorization, or pairwise uncertainty output. Those should
only be added if the low-rank prototype shows useful signal.

### MultiTimescaleNeuralOperator

Current status: implemented with a GRU/SSM fallback and optional Transformer
backbone. The model supports the existing prepared task contract for:

- future-state forecasting
- masked neural reconstruction
- stimulus-to-fMRI response prediction

Known limitation: long-horizon rollout metrics are not yet a first-class
artifact. They should be measured before claiming a rollout advantage.

### StimulusAlignmentAdapter

Current status: partially implemented.

The branch separates stimulus-to-fMRI plumbing from real stimulus evidence by
checking:

- `require_real_stimulus`
- `stimulus_feature_source`
- `stimulus_feature_modalities`
- `stimulus_feature_hash`
- hash-like or synthetic source names

Gap: metadata alone can be spoofed. For claim eligibility, the prepared task
builder should require a source/path field and a non-synthetic/non-hash status.
When a feature path and hash are both available, the hash should be verified
against the loaded feature source or an explicit manifest hash. If this evidence
is absent, stimulus tasks must remain `plumbing_only`.

### UncertaintyHead

Current status: implemented as optional per-time/per-region uncertainty.

Gap: uncertainty calibration is not yet emitted as a dedicated artifact. The
minimal readiness target is an `uncertainty_calibration.csv` with task-level
error/uncertainty summaries when uncertainty outputs exist, or an explicit
unavailable row when they do not.

### EvidenceGate

Current status: paper-mode claim gates exist, and run summaries default
`scientific_claim_allowed=false`. Quarantined tasks are visible in summaries.

Gap: the Pair-Operator path needs an explicit run-local evidence artifact that
states why a run is or is not architecture-claim eligible. Required failures
include:

- missing or failed leakage audit
- empty baseline ranking
- quarantined required task
- non-finite required-task metric
- stimulus-to-fMRI claim requested without real stimulus evidence
- exact competitor reproduction status missing or overstated

The evidence gate must never promote `scientific_claim_allowed` on its own.
`summary.json` remains the source of truth.

## Required Baselines and Labels

The local runner labels must remain explicit:

- `linear_ridge`, `autoregressive_ridge`, `train_mean`, and `persistence` are
  local executable baselines.
- `brainvista_style` is a local clean-room approximation, not exact BrainVista.
- `tribe_style` or `tribe_style_clean_room` is a local clean-room approximation
  when present, not exact TRIBE v2.
- `pair_operator` is the local NeuroTwin Pair-Operator baseline/ablation.

Exact upstream reproduction claims require explicit upstream code/weights and
license/provenance review.

## Required Ablations

The architecture experiment should expose these ablations:

1. current NeuroTwin translator
2. Pair-Operator without pair state
3. Pair-Operator with low-rank pair state
4. Pair-Operator with pair state plus stimulus adapter
5. Pair-Operator with pair state plus uncertainty head
6. Pair-Operator full

The branch currently supports the underlying model flags. It still needs a
cluster-friendly ablation config and a `pair_operator_ablation.csv` artifact.

## A100 Readiness Criteria

Do not run the full 6-GPU job before a 1-GPU debug run passes. A debug run is
ready only when:

- local CPU smoke passes
- Pair-Operator forward paths produce finite metrics
- hash-only stimulus evidence is claim-ineligible
- real-stimulus metadata without a verifiable source remains claim-ineligible
- non-finite required tasks are quarantined and make the evidence gate fail
- baseline ranking is nonempty or explicitly unavailable
- model card and diagnostic report list evidence status and failures

## Historical A100 Command Plan

These commands are retained for reproducibility of the superseded Pair-Operator
branch. They are not the main NFC launch path. The generic Slurm wrapper
`scripts/slurm/train_a100.sh` requests six A100 GPUs. Use the inner launcher for
one-GPU debug jobs, or submit a one-GPU Slurm wrapper around the same inner
command if the cluster requires all GPU work to be queued.

### Historical 1x A100 Pair-Operator Debug

Run this before any full DDP job:

```bash
export RUN_ROOT=/cluster/scratch/$USER/neurotwin_pair_operator_debug
mkdir -p "$RUN_ROOT"
PYTHON_BIN=python3 bash scripts/slurm/_train_a100_inner.sh \
  configs/train/algonauts_pair_operator_debug.yaml \
  "$RUN_ROOT" \
  1
```

Docker fallback:

```bash
GPU_COUNT=1 \
NPROC_PER_NODE=1 \
HOST_GPU_IDS=0 \
CONTAINER_CUDA_VISIBLE_DEVICES=0 \
A100_CONFIG_TEMPLATE=configs/train/algonauts_pair_operator_debug.yaml \
A100_RUN_ID=algonauts_pair_operator_debug \
bash scripts/run_docker_6gpu.sh /cluster/scratch/$USER/neurotwin_pair_operator_debug
```

### Historical 6x A100 Pair-Operator Ablation DDP

Only run this after the 1x debug run passes:

```bash
export RUN_ROOT=/cluster/scratch/$USER/neurotwin_pair_operator_full
mkdir -p "$RUN_ROOT"
sbatch --ntasks-per-node=6 --gres=gpu:a100:6 \
  scripts/slurm/train_a100.sh \
  configs/train/algonauts_pair_operator_full.yaml
```

Docker fallback:

```bash
GPU_COUNT=6 \
NPROC_PER_NODE=6 \
HOST_GPU_IDS=0,1,2,3,4,5 \
CONTAINER_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
A100_CONFIG_TEMPLATE=configs/train/algonauts_pair_operator_full.yaml \
A100_RUN_ID=algonauts_pair_operator_full \
bash scripts/run_docker_6gpu.sh /cluster/scratch/$USER/neurotwin_pair_operator_full
```

Do not use 7 or 8 GPUs for this handoff. The expected full run is exactly six
processes with `torchrun --standalone --nproc_per_node=6`.

### 1x A100 Ablation Array

Queue one independent one-GPU job per ablation variant. Materialize the selected
variant from `configs/train/algonauts_pair_operator_ablation_array.yaml` into a
normal train config before launch, then run:

```bash
export RUN_ROOT=/cluster/scratch/$USER/neurotwin_pair_operator_ablation
mkdir -p "$RUN_ROOT"
PYTHON_BIN=python3 bash scripts/slurm/_train_a100_inner.sh \
  /cluster/scratch/$USER/materialized_configs/<variant>.yaml \
  "$RUN_ROOT" \
  1
```

## Current Gap Checklist

- Add stricter stimulus evidence validation with source/path/status fields.
- Add tests for `require_real_stimulus=true` with missing hash/source.
- Add an explicit required-task gate so quarantined core tasks block claim
  eligibility.
- Add Pair-Operator full and ablation-array configs.
- Emit run-local diagnostic artifacts:
  - `evidence_gate.json`
  - `diagnostic_report.md`
  - `pair_operator_ablation.csv`
  - `uncertainty_calibration.csv`
- Prepare exact 1x A100 debug, 6x A100 DDP, and 1-GPU ablation-array commands.

## Success and Failure Interpretation

This no longer becomes the model paper by itself. Pair-Operator now supports the
NFC paper only if it clarifies whether low-rank relational state helps the full
Neural Field Compiler under leakage-safe held-out evaluation.

If Pair-Operator loses to ridge and BrainVista-style approximations, report the
failure directly. The result remains useful as leakage-gated infrastructure and
evidence hygiene, but not as a model-superiority claim.
