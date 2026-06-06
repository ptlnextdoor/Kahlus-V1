# Pair-Operator Readiness Report

Status: locally hardened for 1x A100 debug preparation
Date: 2026-06-04

## What Was Hardened

- Pair-Operator now has active low-rank pair state that changes predictions.
- 1000-parcel fMRI forward shape is covered without dense `N x N` memory
  materialization for pair confidence.
- Configs expose `use_pair_state`, `pair_rank`, `pair_top_k`,
  `network_blocks`, and `pair_confidence_max_parcels`.
- Stimulus claim eligibility requires local source artifact verification; file
  paths and `file://` URIs are hash-checked against declared hashes.
- Transcript hashes, synthetic embeddings, self-attested embedding hashes, hash
  mismatches, and missing artifacts fail closed as plumbing-only.
- Non-finite loss and non-finite gradients skip optimizer steps and quarantine
  the task.
- Quarantined required tasks force claim disallowance, and aggregate metrics
  ignore NaN values while reporting quarantines.
- Report/model-card commands remain read-only; final evidence sidecars are
  written only by explicit finalization.
- Final baseline ranking validity uses structured `prepared_baseline_suite.json`,
  not CSV scanning.
- Local executable baselines now include `tribe_style_clean_room`,
  `brainvista_style`, `pair_operator`, and `pair_operator_no_pair_state`.
- BrainVista-style baseline uses causal history-only stimulus features by
  default.
- Pair-Operator uncertainty emits finite smoke calibration rows via
  `uncertainty_calibration.csv`.
- A100 launchers require materialized absolute config paths and persistent
  absolute run roots.
- A print-only Pair-Operator A100 command helper and ablation materializer were
  added.

## What Remains Unproven

- No real Algonauts/CNeuroMod A100 metrics have been run.
- No model-superiority, SOTA, or NeurIPS-quality claim is supported yet.
- 6x DDP behavior is script-ready but not cluster-execution-proven.
- Pair uncertainty is diagnostic only; calibration quality is unknown.
- BrainVista-style and TRIBE-style lanes are local approximations, not exact
  upstream reproductions.

## A100 Readiness

1x A100 debug ready: yes, subject to real prepared manifests and CUDA preflight.

6x A100 DDP ready: script-ready, but must follow a successful 1x debug run. Do
not treat it as execution-proven until the first remote DDP job completes with
final evidence sidecars.

## Exact Commands

Set paths first:

```bash
export PREPARED_ROOT=/abs/prepared/algonauts_cneuromod
export RUN_ROOT=/abs/persistent/neurotwin_pair_operator
export CONFIG_ROOT=/abs/persistent/neurotwin_pair_operator/configs
export PHASE1_EVAL_DIR=/abs/persistent/neurotwin_pair_operator/eval/phase1_paper_mode
```

Print the no-submit command sheet:

```bash
bash scripts/print_pair_operator_a100_commands.sh
```

1x A100 debug command:

```bash
mkdir -p "$CONFIG_ROOT" "$RUN_ROOT/runs"
PYTHONPATH=src python3 -m neurotwin.cli cluster materialize-config \
  --template "$(pwd)/configs/train/algonauts_pair_operator_debug.yaml" \
  --prepared-root "$PREPARED_ROOT" \
  --out "$CONFIG_ROOT/algonauts_pair_operator_debug.materialized.yaml"

PYTHON_BIN=python3 \
A100_PAPER_MODE_EVAL_DIR="$PHASE1_EVAL_DIR" \
A100_RUN_PAPER_MODE_IN_FULL=0 \
bash "$(pwd)/scripts/slurm/_train_a100_inner.sh" \
  "$CONFIG_ROOT/algonauts_pair_operator_debug.materialized.yaml" \
  "$RUN_ROOT/runs" \
  1
```

6x A100 full DDP command:

```bash
PYTHONPATH=src python3 -m neurotwin.cli cluster materialize-config \
  --template "$(pwd)/configs/train/algonauts_pair_operator_full.yaml" \
  --prepared-root "$PREPARED_ROOT" \
  --out "$CONFIG_ROOT/algonauts_pair_operator_full.materialized.yaml"

sbatch --ntasks-per-node=6 --gres=gpu:a100:6 \
  --export=ALL,RUN_ROOT="$RUN_ROOT/runs",A100_PAPER_MODE_EVAL_DIR="$PHASE1_EVAL_DIR",A100_RUN_PAPER_MODE_IN_FULL=0 \
  "$(pwd)/scripts/slurm/train_a100.sh" \
  "$CONFIG_ROOT/algonauts_pair_operator_full.materialized.yaml"
```

1-GPU-per-ablation array command:

```bash
python3 scripts/materialize_pair_operator_ablation_configs.py \
  --template "$(pwd)/configs/train/algonauts_pair_operator_ablation_array.yaml" \
  --prepared-root "$PREPARED_ROOT" \
  --out-dir "$CONFIG_ROOT/pair_operator_ablation"

for CONFIG in "$CONFIG_ROOT/pair_operator_ablation"/*.materialized.yaml; do
  sbatch --ntasks-per-node=1 --gres=gpu:a100:1 \
    --export=ALL,RUN_ROOT="$RUN_ROOT/runs",A100_PAPER_MODE_EVAL_DIR="$PHASE1_EVAL_DIR",A100_RUN_PAPER_MODE_IN_FULL=0 \
    "$(pwd)/scripts/slurm/train_a100.sh" \
    "$CONFIG"
done
```

## Artifacts To Inspect

- `summary.json`
- `metrics.csv`
- `metrics.jsonl`
- `eval_audit.json`
- `prepared_baseline_suite.json`
- `seed_aggregate.csv`
- `baseline_failures.json`
- `paper_mode_gate.json`
- `evidence_gate.json`
- `diagnostic_report.md`
- `EEG_MODEL_CARD.md`
- `pair_operator_ablation.csv`
- `uncertainty_calibration.csv`
- `checkpoint_manifest.json`

For stimulus-to-fMRI claims, inspect `stimulus_evidence` in `summary.json` and
the model card before reading any metric as paper evidence.

## Expected Failure Modes

- Real stimulus artifacts are absent, hash mismatched, or only transcript-hash
  derived.
- Pair-Operator beats weak baselines but not `brainvista_style` or the no-pair
  ablation.
- A required task quarantines due to non-finite loss, gradient, or metric.
- Structured baseline suite is missing, unavailable, or malformed.
- Final evidence gate fails despite training completing.
- 6x DDP exposes a distributed-only bug not seen in 1x debug.
- Pair uncertainty exists but calibration metrics are poor or unstable.

## Architecture-Paper Success Criteria

- Real prepared Algonauts/CNeuroMod manifests.
- Leakage audit passes under claim-eligible splits.
- Stimulus features are source-artifact verified for stimulus claims.
- Required tasks finish without quarantine.
- Structured baseline rankings exist for all required tasks.
- Pair-Operator improves at least one core fMRI/stimulus task versus strong
  baselines or the no-pair ablation under the audited split.
- Final model card reports split policy, leakage audit, identity risk, stimulus
  provenance, uncertainty diagnostics, baseline status, and claim gate.
- Paper language avoids first, SOTA, and clinical claims.

## Fallback To Track A

Fall back to the Track A reproducibility paper if the architecture result is not
clean. The fallback criteria are:

- no-pair ablation matches or beats Pair-Operator
- Pair-Operator only wins on invalid or ambiguous splits
- stimulus evidence remains plumbing-only
- final evidence gate fails
- structured baseline ranking is unavailable
- pair state adds complexity without audited task benefit

Track A should stay separate: leakage demos, identity probes, model-card
reporting, and executable claim gates are paper-usable even if Pair-Operator is
not.

## Local Validation Snapshot

Completed locally:

```text
PYTHONPATH=src python3 -m unittest discover -s tests -v
249 tests passed, 2 skipped
PYTHONPATH=src python3 -m neurotwin.cli doctor
passed with expected local CUDA/data-root warnings
bash scripts/run_smoke.sh /tmp/neurotwin_pair_operator_goal_smoke
smoke_status=completed
git diff --check
passed
graphify update .
passed
```
