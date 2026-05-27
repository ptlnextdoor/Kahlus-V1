#!/usr/bin/env bash
#SBATCH --job-name=neurotwin-train-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

if (($# != 1)); then
  echo "usage: sbatch scripts/slurm/train_a100.sh <absolute-path-config-or-materialized-config>" >&2
  echo "Refusing to run the generic placeholder config on A100." >&2
  exit 2
fi

CONFIG=$1
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -z "${RUN_ROOT:-}" ]]; then
  echo "RUN_ROOT must be set to a persistent absolute path before A100 launch." >&2
  exit 2
fi
if [[ "$RUN_ROOT" != /* ]]; then
  echo "RUN_ROOT must be absolute and persistent, got: $RUN_ROOT" >&2
  exit 2
fi

NPROC=${SLURM_NTASKS_PER_NODE:-4}

export PYTHONPATH="${PYTHONPATH:-}:src"
export TOKENIZERS_PARALLELISM=false

"$PYTHON_BIN" -m neurotwin.cli doctor
"$PYTHON_BIN" -m neurotwin.cli cluster preflight \
  --config "$CONFIG" \
  --run-root "$RUN_ROOT" \
  --require-cuda \
  --require-prepared-windows
"$PYTHON_BIN" -m neurotwin.cli train --dry-run --config "$CONFIG"

torchrun --standalone --nproc_per_node="$NPROC" \
  -m neurotwin.cli train --config "$CONFIG" --run-root "$RUN_ROOT"
