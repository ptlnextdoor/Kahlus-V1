#!/usr/bin/env bash
#SBATCH --job-name=neurotwin-eval-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

if (($# != 1)); then
  echo "usage: sbatch scripts/slurm/eval_a100.sh <run-dir>" >&2
  echo "Refusing to run default/synthetic eval on A100." >&2
  exit 2
fi

RUN_DIR=$1
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ ! -d "$RUN_DIR" ]]; then
  echo "RUN_DIR does not exist: $RUN_DIR" >&2
  exit 2
fi

export PYTHONPATH="${PYTHONPATH:-}:src"

"$PYTHON_BIN" -m neurotwin.cli report --run-dir "$RUN_DIR"
