#!/usr/bin/env bash
#SBATCH --job-name=neurotwin-sweep-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

if (($# == 0)); then
  echo "usage: sbatch scripts/slurm/sweep_a100.sh <config> [<config> ...]" >&2
  echo "Refusing to sweep placeholder/default A100 configs." >&2
  exit 2
fi
if [[ -z "${RUN_ROOT:-}" ]]; then
  echo "RUN_ROOT must be set to a persistent absolute path before A100 sweep." >&2
  exit 2
fi
if [[ "$RUN_ROOT" != /* ]]; then
  echo "RUN_ROOT must be absolute and persistent, got: $RUN_ROOT" >&2
  exit 2
fi

CONFIGS=("$@")

for CONFIG in "${CONFIGS[@]}"; do
  sbatch --export=ALL,RUN_ROOT="$RUN_ROOT" scripts/slurm/train_a100.sh "$CONFIG"
done
