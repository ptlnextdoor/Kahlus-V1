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

CONFIG=${1:-configs/train/neurotwin_v1_a100.yaml}
RUN_ROOT=${RUN_ROOT:-runs}
NPROC=${SLURM_NTASKS_PER_NODE:-4}

export PYTHONPATH="${PYTHONPATH:-}:src"
export TOKENIZERS_PARALLELISM=false

python -m neurotwin.cli doctor
python -m neurotwin.cli train --dry-run --config "$CONFIG"

torchrun --standalone --nproc_per_node="$NPROC" \
  -m neurotwin.cli train --config "$CONFIG" --run-root "$RUN_ROOT"
