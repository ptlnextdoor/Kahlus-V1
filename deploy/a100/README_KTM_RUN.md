# Kahlus v3 KTM Sprint 3D A100 Validation — Run Book

Exact, runnable commands for the synthetic KTM Sprint 3D point-objective validation runner.
**Synthetic only.** No MOABB, no real data: **the MOABB audit is not applicable for this synthetic
KTM runner; synthetic split/data-card checks are used instead.** No A100 is run by extracting this
package — run the commands below deliberately.

## 1. Checksum verification

```bash
sha256sum -c SHA256SUMS
```

## 2. Install

```bash
python -m pip install -e .
# or a pinned conda env:
conda env create -f environment-ktm-a100.yml && conda activate kahlus-ktm-a100
```

## 3. CPU smoke (no GPU)

```bash
bash scripts/run_ktm_smoke.sh outputs/smoke      # prints smoke_status=completed
```
Produces `outputs/smoke/{metrics.json,baseline_table.json,baseline_table.csv,evidence_gate.json,
model_card.json,data_card.json,run_config.json,failure_reasons.json,environment.json}` and a
`checkpoints/` dir.

## 4. Docker GPU preflight (prove visible GPU count)

Single GPU:
```bash
docker run --rm --gpus "\"device=0\"" --ipc=host --shm-size=64g \
  -e GPU_COUNT=1 -e CUDA_VISIBLE_DEVICES=0 -e DOCKER_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  -v "$PWD":/workspace/repo -w /workspace/repo pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash -lc 'python -m pip install -e . && GPU_COUNT=1 python scripts/docker_gpu_preflight.py /tmp/gpu_preflight.json'
```
7 GPUs: `--gpus "\"device=0,1,2,3,4,5,6\""`, `GPU_COUNT=7`, `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6`.
8 GPUs: `--gpus "\"device=0,1,2,3,4,5,6,7\""`, `GPU_COUNT=8`, `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`.
Use the actual selected GPU count in every later report and evidence audit.

## 5. Docker Sprint 3D point-objective validation

7-GPU:
```bash
KTM_CONFIG=configs/train/ktm_recovery_point_objective.yaml \
GPU_COUNT=7 HOST_GPU_IDS=0,1,2,3,4,5,6 bash scripts/run_docker_ktm.sh /raid/scratch/$USER/kahlus-ktm-<short_sha>
```
8-GPU:
```bash
KTM_CONFIG=configs/train/ktm_recovery_point_objective.yaml \
GPU_COUNT=8 HOST_GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/run_docker_ktm.sh /raid/scratch/$USER/kahlus-ktm-<short_sha>
```
The launcher writes `docker_run.env`, runs `scripts/docker_ktm_inner.sh` (preflight → CPU smoke →
`torchrun --standalone --nproc_per_node=$GPU_COUNT` with
`configs/train/ktm_recovery_point_objective.yaml`), and tees a log under `$PERSISTENT_ROOT/logs/`.
`configs/train/ktm_recovery_capacity_smoke.yaml` is packaged for follow-up diagnostics only; do not
use it for the first Sprint 3D A100 validation.

## 6. Direct torchrun (inside a configured env)

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/run_ktm_train.py --config configs/train/ktm_recovery_point_objective.yaml \
  --out-dir /abs/persistent/runs/ktm_micro_sweep --mode ddp
```
Use `--nproc_per_node=7` for 7 GPUs. Each rank picks `cuda:LOCAL_RANK`; rank-0 writes the bundle.

## 7. Slurm

```bash
RUN_ROOT=/abs/persistent sbatch scripts/slurm/train_ktm_a100.sh \
  /abs/path/configs/train/ktm_recovery_point_objective.yaml
```
Default header requests `gpu:a100:8` / `--ntasks-per-node=8`; edit to 7 for an honest 7-GPU run.

## 8. Evidence bundle (after a run)

```bash
python scripts/package_ktm_evidence_bundle.py \
  /abs/persistent outputs/kahlus-ktm-a100-results-<short_sha>-evidence.zip \
  kahlus-ktm-a100-results-<short_sha>-evidence . "$(cat COMMIT_HASH.txt)"
```
Includes the run bundle JSON/CSV + `environment.json` + `gpu_preflight.json` + the progress/failure
logs (`progress.jsonl`, `run_status.json`, `failure_report.json`) + logs. **Run the packager even if
the run failed** — it captures partial results + the failure trail for debugging. **Excludes**
`checkpoint*.pt`, `*.pem`, `*.key`, `.env*`, passwords, API keys, W&B tokens. Checkpoints stay on the
cluster.

Audit returned evidence with the actual selected GPU count:

```bash
PYTHONPATH=src python3 scripts/audit_ktm_a100_evidence.py \
  --evidence outputs/kahlus-ktm-a100-results-<short_sha>-evidence.zip \
  --out-dir outputs/ktm-a100-audit-<short_sha> \
  --expected-gpus 8
```

Use `--expected-gpus 7` for an honest 7xA100 run.

## Claims

`synthetic_ktm_training_harness` may pass. `synthetic_ktm_recovery` stays blocked unless KTM beats
baselines under locked metrics. No real-data / clinical / consciousness / Orch-OR / model-superiority
claim.
