#!/usr/bin/env bash
# Inner slurm/direct launcher for the Kahlus v3 KTM synthetic micro-sweep (SYNTHETIC ONLY).
# No MOABB prepare, no manifest extraction, no eval-audit, no run-finalize: the harness writes its
# own self-contained bundle. torchrun sets LOCAL_RANK/RANK/WORLD_SIZE; the code picks cuda:LOCAL_RANK.
set -euo pipefail

if (($# < 2 || $# > 3)); then
  echo "usage: scripts/slurm/_train_ktm_a100_inner.sh <config> <run-root> [nproc]" >&2
  exit 2
fi

CONFIG=$1
RUN_ROOT=$2
NPROC=${3:-${SLURM_NTASKS_PER_NODE:-1}}
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_OUT="$RUN_ROOT/runs/ktm_micro_sweep"
mkdir -p "$RUN_OUT"

"$PYTHON_BIN" -m neurotwin.cli doctor || true

# GPU preflight is best-effort here (only meaningful inside a CUDA context); never blocks the run.
if [[ -n "${GPU_COUNT:-}" ]]; then
  GPU_COUNT="$GPU_COUNT" "$PYTHON_BIN" scripts/docker_gpu_preflight.py "$RUN_ROOT/gpu_preflight.json" || true
fi

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  scripts/run_ktm_train.py --config "$CONFIG" --out-dir "$RUN_OUT" --mode ddp

cp "$RUN_ROOT/gpu_preflight.json" "$RUN_OUT/gpu_preflight.json" 2>/dev/null || true
echo "ktm_micro_sweep_done out_dir=$RUN_OUT nproc=$NPROC"
