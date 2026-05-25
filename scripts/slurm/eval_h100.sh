#!/usr/bin/env bash
#SBATCH --job-name=neurotwin-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

SUITE=${1:-neural_translation_v1}
RUN_DIR=${2:-}

export PYTHONPATH="${PYTHONPATH:-}:src"
python -m neurotwin.cli doctor

if [ -n "$RUN_DIR" ]; then
  python -m neurotwin.cli report --run-dir "$RUN_DIR"
else
  python -m neurotwin.cli eval --suite "$SUITE"
fi
