# NeuroTwin Agent Deploy Instructions

These instructions are for an automated deployment agent running on the A100 cluster. The goal is to run the single-node 6x A100 Docker path when six GPUs are allocated, and to stop rather than silently falling back to one GPU.

Before the full 6-GPU run, submit the Phase 1 evidence lane as separate one-GPU jobs. Queue the paper-mode MOABB seeds 0/1/2 job, leakage-demo seeds 0/1/2 job, identity-probe seeds 0/1/2 job, and model-card/artifact job at the same time when Chapman scheduling allows. If the scheduler serializes them, they must still be independent one-GPU jobs and must finish before the full DDP job starts. Do not reserve six GPUs for Phase 1.

## Inputs

- Extracted runner directory: current working directory.
- Persistent root: `/raid/scratch/$USER/neurotwin-<short_sha>`.
- `GPU_COUNT=6`.
- `HOST_GPU_IDS=0,1,2,3,4,5` unless the scheduler assigns different host GPU ids.
- `CONTAINER_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5` inside the container.
- Docker image: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`.

## Verify The Runner

```bash
cat COMMIT_HASH.txt
sha256sum -c SHA256SUMS
```

## Probe Docker GPU Access

Run this before launching training:

```bash
HOST_GPU_IDS=0,1,2,3,4,5
docker run --rm --gpus "\"device=${HOST_GPU_IDS}\"" \
  --ipc=host --shm-size=64g \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash -lc 'nvidia-smi && python - <<'"'"'PY'"'"'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 6
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY'
```

If `device_count` is not exactly `6`, stop and report the scheduler allocation. Do not run the full handoff as a one-GPU job unless explicitly asked to do a diagnostic run.

## Optional Image Build

`Dockerfile.a100` is a dependency/runtime image helper. It does not hide source code; the runner still contains runtime Python source required to execute.

```bash
docker build -f Dockerfile.a100 -t neurotwin-a100-runner:local .
export DOCKER_IMAGE=neurotwin-a100-runner:local
```

## Full 6-GPU Run

The default handoff command runs the short infrastructure smoke template. For the full MOABB A100 training lane, select the canonical long template before launch:

```bash
export HOST_GPU_IDS=0,1,2,3,4,5
export GPU_COUNT=6
export NPROC_PER_NODE=6
export CONTAINER_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export A100_CONFIG_TEMPLATE=configs/train/moabb_a100.yaml
export A100_RUN_ID=moabb_a100
export PERSISTENT_ROOT=/raid/scratch/$USER/neurotwin-<short_sha>
bash scripts/run_docker_6gpu.sh "$PERSISTENT_ROOT"
```

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

Inside Docker, the selected host GPUs are addressed as `cuda:0` through `cuda:5`. The training path supports single-node DDP through `torchrun`, `LOCAL_RANK`, `RANK`, `WORLD_SIZE`, `torch.cuda.set_device(local_rank)`, and PyTorch `DistributedDataParallel` wrapping.

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
torchrun --standalone --nproc_per_node=6 -m neurotwin.cli train ...
python -m neurotwin.cli run finalize --run-dir "$RUN_DIR" --paper-mode-dir "$A100_PAPER_MODE_EVAL_DIR" ...
bash scripts/package_a100_evidence_bundle.sh "$PERSISTENT_ROOT" outputs
```

For non-smoke run ids, the helper consumes existing Phase 1 artifacts from `A100_PAPER_MODE_EVAL_DIR` when `paper_mode_gate.json` passed. If Phase 1 artifacts are missing, it writes a `paper_mode_artifacts_unavailable` marker and does not silently run paper-mode inside the six-GPU allocation. Only set `A100_RUN_PAPER_MODE_IN_FULL=1` to run paper-mode inside the full allocation.

## One-GPU Diagnostic Only

Use this only to debug Docker/CUDA visibility when six GPUs are not available:

```bash
export HOST_GPU_IDS=<host_gpu_id>
export GPU_COUNT=1
export NPROC_PER_NODE=1
export CONTAINER_CUDA_VISIBLE_DEVICES=0
export PERSISTENT_ROOT=/raid/scratch/$USER/neurotwin-<short_sha>
bash scripts/run_docker_6gpu.sh "$PERSISTENT_ROOT"
```

This launches `torchrun --standalone --nproc_per_node=1` and is not the requested 6-GPU handoff run.

## Expected Evidence

After success, send back the evidence zip written under `outputs/`. It should include summaries, metrics, tables, figures, prepared manifests/audits, `run/gpu_preflight.json`, `run/docker_run.env`, the current Docker log, `COMMIT_HASH.txt`, `README_HANDOFF.md`, `handoff-SHA256SUMS`, and `README_SEND_TO_FRIEND.md`. It must not include checkpoints, secrets, private keys, `.env*`, raw arrays, or the runner tarball.
