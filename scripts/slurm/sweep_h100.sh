#!/usr/bin/env bash
#SBATCH --job-name=neurotwin-sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"

for CONFIG in configs/train/moabb_h100.yaml configs/train/neurotwin_v1_h100.yaml; do
  python -m neurotwin.cli train --dry-run --config "$CONFIG"
  torchrun --standalone --nproc_per_node="${SLURM_NTASKS_PER_NODE:-8}" \
    -m neurotwin.cli train --config "$CONFIG"
done
