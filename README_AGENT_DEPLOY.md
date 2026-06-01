# NeuroTwin Agent Deploy Instructions

These instructions are for an automated deployment agent running on the A100 cluster. The goal is to use the 6-GPU Docker path when six CUDA devices are visible, and to stop rather than silently falling back to one GPU.

## Inputs

- Extracted runner directory: current working directory.
- Persistent root: `/raid/scratch/$USER/neurotwin-<short_sha>`.
- Target GPU count: `6`.
- Docker image base: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`.

## Verify The Runner

```bash
cat COMMIT_HASH.txt
sha256sum -c SHA256SUMS
```

## Probe Docker GPU Access

Run this before launching training:

```bash
docker run --rm --gpus all --ipc=host \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash -lc 'nvidia-smi && python - <<PY
import torch
print("cuda_available=", torch.cuda.is_available())
print("device_count=", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY'
```

If `device_count` is less than `6`, stop and report the scheduler allocation. Do not run the full handoff as a one-GPU job unless explicitly asked to do a diagnostic run.

## Optional Image Build

The runner can execute directly from the mounted directory, or the agent can build a local image first:

```bash
docker build -f Dockerfile.a100 -t neurotwin-a100-runner:local .
```

The standard launcher still uses `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` by default. To use the locally built image:

```bash
export DOCKER_IMAGE=neurotwin-a100-runner:local
```

## Full 6-GPU Run

```bash
export PERSISTENT_ROOT=/raid/scratch/$USER/neurotwin-<short_sha>
TARGET_GPUS=6 bash scripts/run_docker_6gpu.sh "$PERSISTENT_ROOT" all
```

The launcher uses `docker run --gpus all --ipc=host`, probes CUDA inside Docker, writes `$PERSISTENT_ROOT/gpu_preflight.json`, and refuses to start training unless at least six CUDA devices are visible.

The launcher then runs:

```bash
python -m pip install -e '.[moabb,cluster]'
bash scripts/run_smoke.sh outputs/smoke
bash scripts/prepare_moabb_benchmark.sh "$PERSISTENT_ROOT/prepared/moabb_benchmark"
python -m neurotwin.cli eval audit ...
python -m neurotwin.cli cluster materialize-config ...
python -m neurotwin.cli cluster preflight ...
torchrun --standalone --nproc_per_node=6 -m neurotwin.cli train ...
python -m neurotwin.cli report ...
bash scripts/package_a100_evidence_bundle.sh "$PERSISTENT_ROOT" outputs
```

## One-GPU Diagnostic Only

Use this only to debug Docker/CUDA visibility when six GPUs are not available:

```bash
export PERSISTENT_ROOT=/raid/scratch/$USER/neurotwin-<short_sha>
ALLOW_FEWER_GPUS=1 TARGET_GPUS=1 bash scripts/run_docker_6gpu.sh "$PERSISTENT_ROOT" all
```

This is not the requested 6-GPU handoff run.

## Expected Evidence

After success, send back the evidence zip written under `outputs/`. It should include summaries, metrics, tables, figures, prepared manifests/audits, logs, `COMMIT_HASH.txt`, `README_HANDOFF.md`, `handoff-SHA256SUMS`, and `README_SEND_TO_FRIEND.md`. It must not include checkpoints, secrets, private keys, `.env*`, raw arrays, or the runner tarball.
