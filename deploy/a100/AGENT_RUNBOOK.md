# AGENT RUNBOOK — Kahlus v3 KTM Sprint 3D A100 Validation (read this first)

**You are an automation agent running on an A100 cluster node.** This package is a self-contained,
checksummed runner for a **synthetic** Kahlus v3 KTM (Kahlus Transition Model) Sprint 3D validation.
Follow this runbook **exactly and in order**. It is a gated leash run: prove GPU visibility first,
then run one tiny point-objective validation. **Do not improvise, do not change code, do not run a
full sweep.**

## What this is (and is not)

- **Is:** infrastructure validation — does the Sprint 3C `traj_profile` / point-objective direction
  run cluster-native under real `torchrun` DDP on A100s, produce a complete evidence bundle, and keep
  its claim gate honest.
- **Is NOT:** a scientific result. Synthetic data only. **No MOABB, no real EEG, no real-data audit.**
  The MOABB prepare/eval-audit/manifest path does not exist here; leakage safety is the synthetic
  Transition Gym's own split checks + `data_card.json`.

## What is in this package

Inside the extracted runner folder `kahlus-ktm-a100-runner-<sha>/`:

- `AGENT_RUNBOOK.md` — this file.
- `README_HANDOFF.md`, `README_RUN.md` — human + command reference (same commands as below).
- `COMMIT_HASH.txt`, `SHA256SUMS`, `BUNDLE_MANIFEST.txt`, `BUNDLE_METADATA.txt` — provenance + checksums.
- `pyproject.toml`, `environment-ktm-a100.yml` — install metadata (lean; no moabb/mne).
- `configs/train/ktm_recovery_point_objective.yaml` (Sprint 3D validation config),
  `configs/train/ktm_recovery_capacity_smoke.yaml` (follow-up diagnostic only),
  `configs/train/ktm_a100_micro.yaml` (legacy micro config), `configs/train/ktm_synthetic_smoke.yaml`.
- `scripts/run_ktm_train.py`, `scripts/run_ktm_failure_analysis.py`, `scripts/run_ktm_smoke.sh`,
  `scripts/docker_gpu_preflight.py`,
  `scripts/docker_ktm_inner.sh`, `scripts/run_docker_ktm.sh`,
  `scripts/slurm/train_ktm_a100.sh`, `scripts/slurm/_train_ktm_a100_inner.sh`,
  `scripts/package_ktm_evidence_bundle.py`.
- `src/neurotwin/` — the package source.

## Hard rules (non-negotiable)

1. **Preflights before training.** Run the single-GPU and selected-GPU GPU-visibility preflights
   first. If
   **either** fails, STOP, collect logs, report, and **do not train**.
   - **GPU-count honesty.** 8 GPUs is the target. 7 GPUs is acceptable if only 7 are honestly
     available. You MUST predeclare the count (`GPU_COUNT=7` or `GPU_COUNT=8`, `HOST_GPU_IDS=...`)
     and **label every output and report as N×A100** — never claim 8×A100 when you ran 7. The evidence
     bundle records the true `world_size` / visible GPU count; do not contradict it.
2. **No full sweep.** Only the tiny Sprint 3D validation with
   `configs/train/ktm_recovery_point_objective.yaml`.
3. **No code changes on the cluster.** Run the scripts as shipped. If something does not work, report
   the error and logs — do not patch source.
4. **Claim discipline.** `synthetic_ktm_training_harness` may pass. `synthetic_ktm_recovery` must stay
   **blocked** unless the comparison is *locked*: baselines train to a **matched optimizer-step
   budget** (the runner sets baseline steps = the KTM's `steps`, not its short default) **and** the
   trained KTM beats the strongest baseline by at least `recovery_margin` (relative MSE) on the
   locked held-out metrics. A lower KTM MSE under an unmatched budget is a budget artifact, not a
   recovery — the runner enforces this; do not override it. No clinical / real-data / consciousness /
   Orch-OR / model-superiority claims.
5. **No secrets in any output.** The evidence bundle excludes checkpoints, keys, `.env*`, tokens.
6. **Persistent root** must be an absolute cluster scratch path, e.g.
   `/raid/scratch/$USER/kahlus-ktm-<sha>` — not `/tmp`, not a home dir, not inside the runner.

## Standard procedure (do these in order)

### 0. Verify the package

```bash
# in the handoff folder (before extracting the runner)
shasum -a 256 -c SHA256SUMS
# extract the runner, then inside the runner folder:
sha256sum -c SHA256SUMS
```
If a checksum fails, STOP — the package is corrupt; request a fresh copy.

> **Interpreter:** use `python3` (the login/cluster host has `python3` only; bare `python` may be
> missing). All commands below already use `python3`.

### 1. (Optional) CPU smoke — proves the harness without a GPU

```bash
python3 -m pip install -e .
bash scripts/run_ktm_smoke.sh outputs/smoke      # expect: smoke_status=completed
```

### 2. Single-GPU Docker GPU-visibility preflight

```bash
docker run --rm --gpus "\"device=0\"" --ipc=host --shm-size=64g \
  -e GPU_COUNT=1 -e CUDA_VISIBLE_DEVICES=0 -e DOCKER_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  -v "$PWD":/workspace/repo -w /workspace/repo pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash -lc 'python3 -m pip install -e . && GPU_COUNT=1 python3 scripts/docker_gpu_preflight.py /tmp/gpu_preflight_1.json'
```
**Pass = exit 0 and `visible_gpu_count == 1`.** If it fails, STOP and report `/tmp/gpu_preflight_1.json` + logs.

### 3. Selected-GPU Docker GPU-visibility preflight

Use 8 GPUs if available:

```bash
docker run --rm --gpus "\"device=0,1,2,3,4,5,6,7\"" --ipc=host --shm-size=64g \
  -e GPU_COUNT=8 -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -e DOCKER_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  -v "$PWD":/workspace/repo -w /workspace/repo pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash -lc 'python3 -m pip install -e . && GPU_COUNT=8 python3 scripts/docker_gpu_preflight.py /tmp/gpu_preflight_8.json'
```
**Pass = exit 0 and `visible_gpu_count == 8`.** If only 7 GPUs are available, predeclare and test 7:

```bash
docker run --rm --gpus "\"device=0,1,2,3,4,5,6\"" --ipc=host --shm-size=64g \
  -e GPU_COUNT=7 -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 -e DOCKER_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  -v "$PWD":/workspace/repo -w /workspace/repo pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash -lc 'python3 -m pip install -e . && GPU_COUNT=7 python3 scripts/docker_gpu_preflight.py /tmp/gpu_preflight_7.json'
```

**Pass = exit 0 and `visible_gpu_count == selected GPU_COUNT`.** If it fails, STOP and report the
preflight JSON + logs. **Do not train.**

### 4. Only if BOTH preflights pass: tiny Sprint 3D point-objective validation

```bash
PERSISTENT_ROOT=/raid/scratch/$USER/kahlus-ktm-<sha>
KTM_CONFIG=configs/train/ktm_recovery_point_objective.yaml \
GPU_COUNT=8 HOST_GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/run_docker_ktm.sh "$PERSISTENT_ROOT"
```
For 7 GPUs, use `GPU_COUNT=7 HOST_GPU_IDS=0,1,2,3,4,5,6` and label every output as 7xA100.
This runs (inside the container): preflight → CPU smoke → `torchrun --standalone --nproc_per_node=N
scripts/run_ktm_train.py --config configs/train/ktm_recovery_point_objective.yaml --mode ddp`.
Rank-0 writes the bundle to `$PERSISTENT_ROOT/runs/ktm_micro_sweep/`.

Slurm alternative (if Docker is not used):
```bash
RUN_ROOT=$PERSISTENT_ROOT sbatch scripts/slurm/train_ktm_a100.sh \
  "$PWD/configs/train/ktm_recovery_point_objective.yaml"
```

### 5. Package the evidence zip

```bash
python3 scripts/package_ktm_evidence_bundle.py \
  "$PERSISTENT_ROOT" outputs/kahlus-ktm-a100-results-<sha>-evidence.zip \
  kahlus-ktm-a100-results-<sha>-evidence . "$(cat COMMIT_HASH.txt)"
```
Send back **only** this evidence zip. It excludes checkpoints and secrets; checkpoints stay on the
cluster.

Audit returned evidence with the actual GPU count, for example:

```bash
PYTHONPATH=src python3 scripts/audit_ktm_a100_evidence.py \
  --evidence outputs/kahlus-ktm-a100-results-<sha>-evidence.zip \
  --out-dir outputs/ktm-a100-audit-<sha> \
  --expected-gpus 8
```

Use `--expected-gpus 7` for an honest 7xA100 run.

## Expected outputs (`$PERSISTENT_ROOT/runs/ktm_micro_sweep/`)

`metrics.json` · `baseline_table.json` · `baseline_table.csv` · `evidence_gate.json` ·
`model_card.json` · `data_card.json` · `run_config.json` · `failure_reasons.json` ·
`environment.json` · `gpu_preflight.json` · `progress.jsonl` · `run_status.json` ·
`checkpoints/` (stays on cluster). On a crash, additionally `failure_report.json`.

## Progress + failure logging (recover partial results, debug accurately)

The run logs incrementally so nothing is lost if it dies at startup, mid-run, or at the end:

- `environment.json` is written **before** training starts (so even a startup crash captures the runtime).
- `progress.jsonl` appends one line per event: `run_started`, each `step` (with loss), each `eval`
  (with val MSE), each `checkpoint`, `nonfinite_skip`, `loss_explosion`, `val_after`,
  `training_error`. Read the last lines to see exactly how far it got.
- `run_status.json` is a one-glance snapshot: `status` (`running` / `training_complete` /
  `completed` / `failed`), `phase` (`setup`/`data`/`model`/`train`/`bundle`), `completed_steps`,
  `total_steps`.
- On any exception, `failure_report.json` records the phase, completed steps, error type + message,
  full traceback, and which partial artifacts/checkpoints exist.
- Checkpoints are saved **incrementally** (best + last every few steps), so a late crash still leaves
  a usable partial model on the cluster.

## Expected claim status (what "success" looks like)

- `evidence_gate.json` → `claim_scope = synthetic_ktm_training_harness`,
  `scientific_claim_allowed = true` (harness readiness only).
- `metrics.json` → `recovery_claim_allowed = false` unless the trained KTM beat the strongest
  baseline by `recovery_margin` under a **matched optimizer-step budget**
  (`ktm_vs_baselines.budget_matched = true`, `comparison_locked = true`). **A decreasing loss — or a
  lower MSE under an unmatched budget — does NOT earn the recovery claim.**
- `environment.json` records torch/CUDA/NCCL versions, visible GPU count + names,
  `CUDA_VISIBLE_DEVICES`, `WORLD_SIZE`.

## If something fails

- Checksum mismatch → corrupt package; request a fresh copy.
- Preflight visible-GPU-count mismatch → GPU exposure problem; report the preflight JSON; do not train.
- Training error → **still run the evidence packager (Step 5)**. It collects `progress.jsonl`,
  `run_status.json`, `failure_report.json`, any partial bundle files, and `$PERSISTENT_ROOT/logs/*`,
  so we can recover partial results and debug the code accurately. Then report. **Do not patch code.**
  Checkpoints stay on the cluster (excluded from the evidence zip) unless explicitly requested.

This is the Sprint 3D CUDA validation. **Leash stays on:** preflights first, one tiny point-objective
run, full evidence bundle, no recovery claim unless earned, no full sweep.
