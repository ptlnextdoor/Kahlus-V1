#!/usr/bin/env bash
#SBATCH --job-name=neurotwin-a100-full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if (($# != 2)); then
  echo "usage: scripts/run_full.sbatch <materialized-config> <run-root>" >&2
  exit 2
fi

CONFIG=$1
RUN_ROOT=$2
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ "$CONFIG" != /* ]]; then
  echo "CONFIG must be a materialized absolute path, got: $CONFIG" >&2
  exit 2
fi
if [[ -z "$REPO_ROOT" ]]; then
  echo "REPO_ROOT or SLURM_SUBMIT_DIR must be set so the Slurm job can find the checkout." >&2
  exit 2
fi
if [[ "$REPO_ROOT" != /* ]]; then
  echo "REPO_ROOT must be absolute, got: $REPO_ROOT" >&2
  exit 2
fi
cd "$REPO_ROOT"
if [[ ! -f src/neurotwin/cli.py || ! -f scripts/train_a100_inner.sh ]]; then
  echo "REPO_ROOT does not look like a NeuroTwin checkout: $REPO_ROOT" >&2
  exit 2
fi

export RUN_ROOT
export PYTHON_BIN="${PYTHON_BIN:-python3}"

bash scripts/train_a100_inner.sh "$CONFIG" "$RUN_ROOT" 1
