#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: PREPARED_ROOT=/abs/prepared RUN_ROOT=/abs/persistent CONFIG_ROOT=/abs/configs PHASE1_EVAL_DIR=/abs/eval bash scripts/print_pair_operator_a100_commands.sh

Prints exact Pair-Operator A100 commands. It does not submit jobs.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

for name in PREPARED_ROOT RUN_ROOT CONFIG_ROOT PHASE1_EVAL_DIR; do
  if [[ -z "${!name:-}" ]]; then
    echo "$name must be set to an absolute path." >&2
    usage
    exit 2
  fi
  if [[ "${!name}" != /* ]]; then
    echo "$name must be absolute, got: ${!name}" >&2
    exit 2
  fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
if [[ "$REPO_ROOT" != /* ]]; then
  echo "REPO_ROOT must be absolute, got: $REPO_ROOT" >&2
  exit 2
fi

DEBUG_CONFIG="$CONFIG_ROOT/algonauts_pair_operator_debug.materialized.yaml"
FULL_CONFIG="$CONFIG_ROOT/algonauts_pair_operator_full.materialized.yaml"
ABLATION_CONFIG_DIR="$CONFIG_ROOT/pair_operator_ablation"
A100_RUN_ROOT="$RUN_ROOT/runs"

cat <<EOF
mkdir -p "$CONFIG_ROOT" "$A100_RUN_ROOT"

PYTHONPATH=src python3 -m neurotwin.cli cluster materialize-config \\
  --template "$REPO_ROOT/configs/train/algonauts_pair_operator_debug.yaml" \\
  --prepared-root "$PREPARED_ROOT" \\
  --out "$DEBUG_CONFIG"

PYTHON_BIN=python3 \\
A100_PAPER_MODE_EVAL_DIR="$PHASE1_EVAL_DIR" \\
A100_RUN_PAPER_MODE_IN_FULL=0 \\
bash "$REPO_ROOT/scripts/slurm/_train_a100_inner.sh" \\
  "$DEBUG_CONFIG" \\
  "$A100_RUN_ROOT" \\
  1

PYTHONPATH=src python3 -m neurotwin.cli cluster materialize-config \\
  --template "$REPO_ROOT/configs/train/algonauts_pair_operator_full.yaml" \\
  --prepared-root "$PREPARED_ROOT" \\
  --out "$FULL_CONFIG"

A100_PAPER_MODE_EVAL_DIR="$PHASE1_EVAL_DIR" \\
A100_RUN_PAPER_MODE_IN_FULL=0 \\
sbatch --ntasks-per-node=6 --gres=gpu:a100:6 \\
  --export=ALL,RUN_ROOT="$A100_RUN_ROOT",A100_PAPER_MODE_EVAL_DIR="$PHASE1_EVAL_DIR",A100_RUN_PAPER_MODE_IN_FULL=0 \\
  "$REPO_ROOT/scripts/slurm/train_a100.sh" \\
  "$FULL_CONFIG"

python3 "$REPO_ROOT/scripts/materialize_pair_operator_ablation_configs.py" \\
  --template "$REPO_ROOT/configs/train/algonauts_pair_operator_ablation_array.yaml" \\
  --prepared-root "$PREPARED_ROOT" \\
  --out-dir "$ABLATION_CONFIG_DIR"

for CONFIG in "$ABLATION_CONFIG_DIR"/*.materialized.yaml; do
  sbatch --ntasks-per-node=1 --gres=gpu:a100:1 \\
    --export=ALL,RUN_ROOT="$A100_RUN_ROOT",A100_PAPER_MODE_EVAL_DIR="$PHASE1_EVAL_DIR",A100_RUN_PAPER_MODE_IN_FULL=0 \\
    "$REPO_ROOT/scripts/slurm/train_a100.sh" \\
    "\$CONFIG"
done
EOF
