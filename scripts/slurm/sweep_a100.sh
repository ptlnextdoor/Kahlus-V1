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

if (($#)); then
  CONFIGS=("$@")
else
  CONFIGS=(configs/train/moabb_a100.yaml)
fi
RUN_ROOT=${RUN_ROOT:-runs}

for CONFIG in "${CONFIGS[@]}"; do
  sbatch --export=ALL,RUN_ROOT="$RUN_ROOT" scripts/slurm/train_a100.sh "$CONFIG"
done
