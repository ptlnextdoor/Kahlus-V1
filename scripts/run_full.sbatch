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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export RUN_ROOT
export PYTHON_BIN="${PYTHON_BIN:-python3}"

bash scripts/slurm/_train_a100_inner.sh "$CONFIG" "$RUN_ROOT" 1
