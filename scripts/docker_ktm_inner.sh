#!/usr/bin/env bash
# In-container runner for the Kahlus v3 KTM synthetic A100 micro-sweep (SYNTHETIC ONLY).
# Runs inside the pytorch CUDA image with the repo mounted at /workspace/repo. Steps:
#   1. install the package        2. GPU preflight (visible==GPU_COUNT)
#   3. CPU smoke sanity           4. DDP micro-sweep via torchrun (nproc==NPROC_PER_NODE)
# NO MOABB, NO real data, NO eval-audit/manifest path. rank-0 writes the output bundle.
set -euo pipefail

: "${PERSISTENT_ROOT:?PERSISTENT_ROOT must be set (absolute, persistent)}"
: "${GPU_COUNT:?GPU_COUNT must be set}"
: "${NPROC_PER_NODE:?NPROC_PER_NODE must be set}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
KTM_CONFIG="${KTM_CONFIG:-configs/train/ktm_a100_micro.yaml}"
PIP_EXTRA="${KTM_PIP_EXTRA:-.[cluster]}"
RUN_OUT="$PERSISTENT_ROOT/runs/ktm_micro_sweep"
mkdir -p "$RUN_OUT"

echo "ktm_inner persistent_root=$PERSISTENT_ROOT gpu_count=$GPU_COUNT nproc=$NPROC_PER_NODE config=$KTM_CONFIG"

"$PYTHON_BIN" -m pip install -e "$PIP_EXTRA"

# Proof Docker sees exactly GPU_COUNT devices; writes the flat env fields for evidence.
GPU_COUNT="$GPU_COUNT" "$PYTHON_BIN" scripts/docker_gpu_preflight.py "$PERSISTENT_ROOT/gpu_preflight.json"

# CPU smoke sanity (cheap; proves the harness + bundle path before touching DDP).
bash scripts/run_ktm_smoke.sh "$PERSISTENT_ROOT/smoke"

# True DDP micro-sweep. torchrun sets LOCAL_RANK/RANK/WORLD_SIZE; the code picks cuda:LOCAL_RANK.
torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
  scripts/run_ktm_train.py --config "$KTM_CONFIG" --out-dir "$RUN_OUT" --mode ddp

# Keep the preflight proof alongside the run bundle for the evidence package.
cp "$PERSISTENT_ROOT/gpu_preflight.json" "$RUN_OUT/gpu_preflight.json" 2>/dev/null || true

echo "ktm_micro_sweep_done out_dir=$RUN_OUT"
