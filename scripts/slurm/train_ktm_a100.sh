#!/usr/bin/env bash
#SBATCH --job-name=kahlus-ktm-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Kahlus v3 KTM synthetic A100 micro-sweep, slurm path (SYNTHETIC ONLY). Defaults to 8 procs;
# override --ntasks-per-node / --gres to 6 for a 6-GPU run. Refuses the generic placeholder config.
set -euo pipefail

if (($# != 1)); then
  echo "usage: RUN_ROOT=/abs/persistent sbatch scripts/slurm/train_ktm_a100.sh <absolute-config>" >&2
  exit 2
fi

CONFIG=$1
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ "$CONFIG" != /* ]]; then
  echo "Config must be an absolute path before A100 launch, got: $CONFIG" >&2
  exit 2
fi
if [[ -z "${RUN_ROOT:-}" ]]; then
  echo "RUN_ROOT must be set to a persistent absolute path before A100 launch." >&2
  exit 2
fi
if [[ "$RUN_ROOT" != /* ]]; then
  echo "RUN_ROOT must be absolute and persistent, got: $RUN_ROOT" >&2
  exit 2
fi

NPROC=${SLURM_NTASKS_PER_NODE:-8}
export PYTHON_BIN

bash scripts/slurm/_train_ktm_a100_inner.sh "$CONFIG" "$RUN_ROOT" "$NPROC"
