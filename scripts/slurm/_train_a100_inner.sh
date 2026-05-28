#!/usr/bin/env bash
set -euo pipefail

if (($# < 2 || $# > 3)); then
  echo "usage: scripts/slurm/_train_a100_inner.sh <config> <run-root> [nproc]" >&2
  exit 2
fi

CONFIG=$1
RUN_ROOT=$2
NPROC=${3:-${SLURM_NTASKS_PER_NODE:-1}}
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config does not exist: $CONFIG" >&2
  exit 2
fi
if [[ -z "$RUN_ROOT" || "$RUN_ROOT" != /* ]]; then
  echo "RUN_ROOT must be an absolute persistent path, got: $RUN_ROOT" >&2
  exit 2
fi
if [[ ! -d "$RUN_ROOT" ]]; then
  echo "RUN_ROOT does not exist: $RUN_ROOT" >&2
  exit 2
fi
if [[ "$NPROC" -lt 1 ]]; then
  echo "nproc must be positive, got: $NPROC" >&2
  exit 2
fi

export PYTHONPATH="${PYTHONPATH:-}:src"
export TOKENIZERS_PARALLELISM=false

RUN_DIR="$("$PYTHON_BIN" - "$CONFIG" "$RUN_ROOT" <<'PY'
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
print(Path(sys.argv[2]) / str(config.get("experiment", "synthetic_debug")))
PY
)"

"$PYTHON_BIN" -m neurotwin.cli doctor
"$PYTHON_BIN" -m neurotwin.cli cluster preflight \
  --config "$CONFIG" \
  --run-root "$RUN_ROOT" \
  --require-cuda \
  --require-prepared-windows
"$PYTHON_BIN" -m neurotwin.cli train --dry-run --config "$CONFIG"

torchrun --standalone --nproc_per_node="$NPROC" \
  -m neurotwin.cli train --config "$CONFIG" --run-root "$RUN_ROOT"

"$PYTHON_BIN" -m neurotwin.cli report --run-dir "$RUN_DIR"
