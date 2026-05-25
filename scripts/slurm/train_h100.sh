#!/usr/bin/env bash
#SBATCH --job-name=neurotwin-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

CONFIG=${1:-configs/train/neurotwin_v1_h100.yaml}
RUN_ROOT=${RUN_ROOT:-runs}

export PYTHONPATH="${PYTHONPATH:-}:src"
export TOKENIZERS_PARALLELISM=false

python -m neurotwin.cli doctor
python -m neurotwin.cli train --dry-run --config "$CONFIG"

torchrun --standalone --nproc_per_node="${SLURM_NTASKS_PER_NODE:-8}" \
  -m neurotwin.cli train --config "$CONFIG" --run-root "$RUN_ROOT"
