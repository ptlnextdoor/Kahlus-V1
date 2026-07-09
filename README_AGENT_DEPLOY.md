# NeuroTwin Agent Deploy Instructions

These instructions are for an automated deployment agent running on the A100 cluster. The goal is to run the single-node 7x A100 Docker path when seven GPUs are allocated, and to stop rather than silently falling back to one GPU.

Krish's partner cluster may expose 7x A100 80GB GPUs. Use exactly 7x A100 80GB GPUs for the full lane, and run inside a `12:00:00` allocation for the deep job. Docker does not extend queue wall time; the scheduler allocation must already be long enough. There is no artificial per-GPU VRAM cap in this runner.

Before the full 7-GPU run, submit the Phase 1 evidence lane as separate one-GPU jobs. Queue the paper-mode MOABB seeds 0/1/2 job, leakage-demo seeds 0/1/2 job, identity-probe seeds 0/1/2 job, and model-card/artifact job at the same time when Chapman scheduling allows. If the scheduler serializes them, they must still be independent one-GPU jobs and must finish before the full DDP job starts. Do not reserve seven GPUs for Phase 1.

## Inputs

- Extracted runner directory: current working directory.
- Persistent root: `/raid/scratch/$USER/neurotwin-<short_sha>`.
- `GPU_COUNT=7`.
- `HOST_GPU_IDS=0,1,2,3,4,5,6` unless the scheduler assigns different host GPU ids.
- `CONTAINER_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6` inside the container.
- Allocation: exactly 7 of the available 7x A100 80GB GPUs, `12:00:00` wall time for the deep lane.
- Docker image: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`.

## Verify The Runner

```bash
cat COMMIT_HASH.txt
sha256sum -c SHA256SUMS
```

## Probe Docker GPU Access

Run this before launching training:

```bash
HOST_GPU_IDS=0,1,2,3,4,5,6
docker run --rm --gpus "\"device=${HOST_GPU_IDS}\"" \
  --ipc=host --shm-size=64g \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash -lc 'nvidia-smi && python - <<'"'"'PY'"'"'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 7
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY'
```

If `device_count` is not exactly `7`, stop and report the scheduler allocation. Do not run the full handoff as a one-GPU job unless explicitly asked to do a diagnostic run.

## Optional Image Build

`Dockerfile.a100` is a dependency/runtime image helper. It does not hide source code; the runner still contains runtime Python source required to execute.

```bash
docker build -f Dockerfile.a100 -t neurotwin-a100-runner:local .
export DOCKER_IMAGE=neurotwin-a100-runner:local
```

## Full 7-GPU Run

The default 7-GPU handoff command runs the full `configs/train/moabb_a100.yaml` lane. Short diagnostics should use `scripts/run_smoke.sh` or a one-GPU diagnostic explicitly. Short diagnostic runs ending in a few hours are normal and do not mean the 12-hour deep lane was exercised.


```bash
export HOST_GPU_IDS=0,1,2,3,4,5,6
export GPU_COUNT=7
export NPROC_PER_NODE=7
export CONTAINER_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export A100_CONFIG_TEMPLATE=configs/train/moabb_a100.yaml
export A100_RUN_ID=moabb_a100
export PERSISTENT_ROOT=/raid/scratch/$USER/neurotwin-<short_sha>
bash scripts/run_docker_7gpu.sh "$PERSISTENT_ROOT"
```

Run the command inside an existing 12-hour allocation. If Chapman supports interactive Slurm allocations, the shape is:

```bash
salloc --nodes=1 --ntasks-per-node=7 --gres=gpu:a100:7 --time=12:00:00 --mem=0
```

If using `sbatch` instead, keep `--time=12:00:00`. The long template uses `configs/train/moabb_a100.yaml` with `steps: 50000` (50,000 configured steps).

The launcher uses:

```bash
docker run --rm --gpus "\"device=${HOST_GPU_IDS}\"" \
  --ipc=host --shm-size=64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e CUDA_VISIBLE_DEVICES="$CONTAINER_CUDA_VISIBLE_DEVICES" \
  -e NCCL_DEBUG=INFO \
  -v "$PWD":/workspace/repo \
  -v "$PERSISTENT_ROOT":"$PERSISTENT_ROOT" \
  -w /workspace/repo \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash scripts/docker_a100_inner.sh
```

The host launcher auto-generates `DOCKER_LOG_PATH`, records it in `$PERSISTENT_ROOT/docker_run.env`, and writes `neurotwin-a100-docker-<generated>.log` under `$PERSISTENT_ROOT/logs`.

Inside Docker, the selected host GPUs are addressed as `cuda:0` through `cuda:6`. The training path supports single-node DDP through `torchrun`, `LOCAL_RANK`, `RANK`, `WORLD_SIZE`, `torch.cuda.set_device(local_rank)`, and PyTorch `DistributedDataParallel` wrapping.

The inner runner executes:

```bash
python -m pip install -e '.[moabb,cluster]'
python scripts/docker_gpu_preflight.py "$PERSISTENT_ROOT/gpu_preflight.json"
bash scripts/run_smoke.sh outputs/smoke
bash scripts/prepare_moabb_benchmark.sh "$PERSISTENT_ROOT/prepared/moabb_benchmark"
python -m neurotwin.cli eval audit ...
python -m neurotwin.cli cluster materialize-config --template "$A100_CONFIG_TEMPLATE" ...
python -m neurotwin.cli cluster preflight ...
copy Phase 1 paper-mode artifacts from "$A100_PAPER_MODE_EVAL_DIR" when paper_mode_gate.json passed
torchrun --standalone --nproc_per_node=7 -m neurotwin.cli train ...
python -m neurotwin.cli run finalize --run-dir "$RUN_DIR" --paper-mode-dir "$A100_PAPER_MODE_EVAL_DIR" ...
bash scripts/package_a100_evidence_bundle.sh "$PERSISTENT_ROOT" outputs
```

For non-smoke run ids, the helper consumes existing Phase 1 artifacts from `A100_PAPER_MODE_EVAL_DIR` when `paper_mode_gate.json` passed. If Phase 1 artifacts are missing, it writes a `paper_mode_artifacts_unavailable` marker and does not silently run paper-mode inside the seven-GPU allocation. Only set `A100_RUN_PAPER_MODE_IN_FULL=1` to run paper-mode inside the full allocation.

## One-GPU Diagnostic Only

Use this only to debug Docker/CUDA visibility when seven GPUs are not available:

```bash
export HOST_GPU_IDS=<host_gpu_id>
export GPU_COUNT=1
export NPROC_PER_NODE=1
export CONTAINER_CUDA_VISIBLE_DEVICES=0
export PERSISTENT_ROOT=/raid/scratch/$USER/neurotwin-<short_sha>
bash scripts/run_docker_7gpu.sh "$PERSISTENT_ROOT"
```

This launches `torchrun --standalone --nproc_per_node=1` and is not the requested 7-GPU handoff run.

## Expected Evidence

After success, send back the evidence zip written under `outputs/`. It should include summaries, metrics, tables, figures, prepared manifests/audits, `run/gpu_preflight.json`, `run/docker_run.env`, the current Docker log, `COMMIT_HASH.txt`, `README_HANDOFF.md`, `handoff-SHA256SUMS`, and `README_SEND_TO_FRIEND.md`. It must not include checkpoints, secrets, private keys, `.env*`, raw arrays, or the runner tarball.
