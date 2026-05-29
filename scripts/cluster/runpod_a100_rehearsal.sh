#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: scripts/cluster/runpod_a100_rehearsal.sh /workspace/neurotwin_data" >&2
}

if (($# != 1)); then
  usage
  exit 2
fi

DATA_ROOT=$1
if [[ "$DATA_ROOT" != /* ]]; then
  echo "RunPod data root must be absolute, got: $DATA_ROOT" >&2
  exit 2
fi
case "$DATA_ROOT" in
  /tmp|/tmp/*|/private/tmp|/private/tmp/*|/var/tmp|/var/tmp/*)
    echo "RunPod data root must not be tmp: $DATA_ROOT" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${PYTHONPATH:-}:src"
export RUNPOD_MAX_BUDGET_USD="${RUNPOD_MAX_BUDGET_USD:-5}"
export RUNPOD_MAX_SECONDS="${RUNPOD_MAX_SECONDS:-}"
export RUNPOD_MAX_GPU_USD_PER_HOUR="${RUNPOD_MAX_GPU_USD_PER_HOUR:-}"

if "$PYTHON_BIN" - <<'PY'
import os
import sys

budget = float(os.environ["RUNPOD_MAX_BUDGET_USD"])
sys.exit(0 if budget <= 5.0 else 1)
PY
then
  :
else
  echo "RUNPOD_MAX_BUDGET_USD must be <= 5 for this rehearsal." >&2
  exit 2
fi

if [[ -z "$RUNPOD_MAX_SECONDS" ]]; then
  RUNPOD_MAX_SECONDS="$("$PYTHON_BIN" - <<'PY'
import os

budget = float(os.environ["RUNPOD_MAX_BUDGET_USD"])
rate = os.environ.get("RUNPOD_MAX_GPU_USD_PER_HOUR")
if rate:
    seconds = int((budget / float(rate)) * 3600)
else:
    seconds = 3600
print(max(300, min(seconds, 3600)))
PY
)"
fi
export RUNPOD_MAX_SECONDS

START_EPOCH="$(date +%s)"
deadline_remaining() {
  local now elapsed remaining
  now="$(date +%s)"
  elapsed=$((now - START_EPOCH))
  remaining=$((RUNPOD_MAX_SECONDS - elapsed))
  if ((remaining <= 0)); then
    echo "RunPod rehearsal budget time exhausted after ${elapsed}s." >&2
    exit 124
  fi
  printf '%s\n' "$remaining"
}

run_limited() {
  local remaining
  remaining="$(deadline_remaining)"
  if command -v timeout >/dev/null 2>&1; then
    timeout "${remaining}s" "$@"
  else
    "$@"
  fi
}

mkdir -p "$DATA_ROOT"
PERSISTENT_ROOT="$(cd "$DATA_ROOT" && pwd)"
export NEUROTWIN_DATA="$PERSISTENT_ROOT"
export MOABB_DATA="$NEUROTWIN_DATA/moabb"
export BIDS_ROOT="$NEUROTWIN_DATA/bids"
export RUN_ROOT="$NEUROTWIN_DATA/runs"
export EXPECTED_WINDOW_COUNT="${EXPECTED_WINDOW_COUNT:-18144}"
export EXPECTED_SPLIT_WINDOWS="${EXPECTED_SPLIT_WINDOWS:-train:12096,val:2016,test:4032}"
mkdir -p logs outputs/configs "$MOABB_DATA" "$BIDS_ROOT" "$RUN_ROOT" "$NEUROTWIN_DATA/prepared"

echo "runpod_budget_usd=$RUNPOD_MAX_BUDGET_USD"
echo "runpod_max_seconds=$RUNPOD_MAX_SECONDS"
echo "neurotwin_data=$NEUROTWIN_DATA"

echo "step=runpod_cuda_check"
run_limited nvidia-smi
run_limited "$PYTHON_BIN" - <<'PY'
import torch

print(f"torch_cuda_available={torch.cuda.is_available()}")
print(f"torch_cuda_device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("CUDA is required for RunPod A100 rehearsal")
name = torch.cuda.get_device_name(0)
print(f"torch_cuda_device_0={name}")
if "A100" not in name:
    raise SystemExit(f"expected an A100-class GPU, got: {name}")
PY

echo "step=smoke"
run_limited bash scripts/run_smoke.sh "$NEUROTWIN_DATA/smoke"

PREPARED_DIR="$NEUROTWIN_DATA/prepared/moabb_benchmark"
echo "step=prepare_moabb_benchmark"
run_limited bash scripts/prepare_moabb_benchmark.sh "$PREPARED_DIR"

CONFIG_PATH="outputs/configs/moabb_a100.runpod.yaml"
echo "step=materialize_runpod_config path=$CONFIG_PATH"
run_limited "$PYTHON_BIN" -m neurotwin.cli cluster materialize-config \
  --template configs/train/moabb_a100_smoke.yaml \
  --prepared-root "$PREPARED_DIR" \
  --out "$CONFIG_PATH"

echo "step=runpod_preflight"
run_limited "$PYTHON_BIN" -m neurotwin.cli cluster preflight \
  --config "$CONFIG_PATH" \
  --run-root "$RUN_ROOT" \
  --require-cuda \
  --require-prepared-windows \
  --expect-window-count "$EXPECTED_WINDOW_COUNT" \
  --expect-split-windows "$EXPECTED_SPLIT_WINDOWS"

echo "step=runpod_fake_slurm_training"
run_limited env \
  SLURM_JOB_ID=runpod-rehearsal \
  SLURM_SUBMIT_DIR="$REPO_ROOT" \
  REPO_ROOT="$REPO_ROOT" \
  bash scripts/run_full.sbatch "$CONFIG_PATH" "$RUN_ROOT"

echo "runpod_rehearsal_passed=True"
