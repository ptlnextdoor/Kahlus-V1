#!/usr/bin/env bash
#SBATCH --job-name=neurotwin-train-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:a100:6
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
if [[ "$CONFIG" != /* ]]; then
  echo "Config must be a materialized absolute path before A100 launch, got: $CONFIG" >&2
  exit 2
fi
if [[ -z "${RUN_ROOT:-}" ]]; then
  echo "RUN_ROOT must be set to a persistent absolute path before A100 launch." >&2
  exit 2
fi
if [[ "$RUN_ROOT" != /* ]]; then
  echo "RUN_ROOT must be absolute and persistent, got: $RUN_ROOT" >&2
  exit 2
fi

NPROC=${SLURM_NTASKS_PER_NODE:-6}
export PYTHON_BIN

bash scripts/slurm/_train_a100_inner.sh "$CONFIG" "$RUN_ROOT" "$NPROC"
