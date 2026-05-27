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

SUITE=${1:-neural_translation_v1}
RUN_DIR=${2:-runs/neurotwin_v1_a100}

export PYTHONPATH="${PYTHONPATH:-}:src"

python -m neurotwin.cli report --run-dir "$RUN_DIR"
python -m neurotwin.cli eval --suite "$SUITE"
