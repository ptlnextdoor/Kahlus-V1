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

eval "$("$PYTHON_BIN" - "$CONFIG" "$RUN_ROOT" <<'PY'
import shlex
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
data = config.get("data") if isinstance(config.get("data"), dict) else {}


def emit(name, value):
    print(f"{name}={shlex.quote(str(value or ''))}")


emit("RUN_DIR", Path(sys.argv[2]) / str(config.get("experiment", "synthetic_debug")))
emit("EVENT_MANIFEST", data.get("event_manifest") or config.get("event_manifest") or "")
emit("SPLIT_MANIFEST", data.get("split_manifest") or config.get("split_manifest") or "")
emit("WINDOW_LENGTH", config.get("window_size", 8))
emit("STRIDE", config.get("stride", 8))
PY
)"

copy_a100_paper_artifacts() {
  local run_dir=$1
  local eval_dir=${A100_PAPER_MODE_EVAL_DIR:-}
  if [[ -z "$eval_dir" ]]; then
    return 0
  fi
  for artifact in \
    prepared_baseline_suite.json \
    seed_aggregate.json \
    seed_aggregate.csv \
    baseline_failures.json \
    paper_mode_gate.json; do
    if [[ -f "$eval_dir/$artifact" ]]; then
      cp "$eval_dir/$artifact" "$run_dir/$artifact"
    fi
  done
}

copy_prepared_eval_audit() {
  local run_dir=$1
  local event_manifest=$2
  if [[ -z "$event_manifest" || ! -f "$event_manifest" ]]; then
    return 0
  fi
  local prepared_dir
  prepared_dir="$(dirname "$event_manifest")"
  if [[ -f "$prepared_dir/eval_audit.json" ]]; then
    cp "$prepared_dir/eval_audit.json" "$run_dir/eval_audit.json"
  fi
}

run_a100_paper_diagnostics() {
  local run_dir=$1
  local event_manifest=$2
  local split_manifest=$3
  local window_length=$4
  local stride=$5
  if [[ ! -f "$event_manifest" || ! -f "$split_manifest" ]]; then
    echo "Skipping paper diagnostics; prepared manifests are missing." >&2
    return 0
  fi
  "$PYTHON_BIN" -m neurotwin.cli eval leakage-demo \
    --seeds 0 1 2 \
    --event-manifest "$event_manifest" \
    --split-manifest "$split_manifest" \
    --window-length "$window_length" \
    --stride "$stride" \
    --out-dir "$run_dir"
  "$PYTHON_BIN" -m neurotwin.cli eval identity-probe \
    --seeds 0 1 2 \
    --event-manifest "$event_manifest" \
    --split-manifest "$split_manifest" \
    --window-length "$window_length" \
    --stride "$stride" \
    --out-dir "$run_dir"
}

PREFLIGHT_ARGS=(
  --config "$CONFIG"
  --run-root "$RUN_ROOT"
  --require-cuda
  --require-prepared-windows
)
if [[ -n "${EXPECTED_WINDOW_COUNT:-}" ]]; then
  PREFLIGHT_ARGS+=(--expect-window-count "$EXPECTED_WINDOW_COUNT")
fi
if [[ -n "${EXPECTED_SPLIT_WINDOWS:-}" ]]; then
  PREFLIGHT_ARGS+=(--expect-split-windows "$EXPECTED_SPLIT_WINDOWS")
fi

"$PYTHON_BIN" -m neurotwin.cli doctor
"$PYTHON_BIN" -m neurotwin.cli cluster preflight "${PREFLIGHT_ARGS[@]}"
"$PYTHON_BIN" -m neurotwin.cli train --dry-run --config "$CONFIG"

torchrun --standalone --nproc_per_node="$NPROC" \
  -m neurotwin.cli train --config "$CONFIG" --run-root "$RUN_ROOT"

copy_a100_paper_artifacts "$RUN_DIR"
copy_prepared_eval_audit "$RUN_DIR" "$EVENT_MANIFEST"
"$PYTHON_BIN" -m neurotwin.cli report --run-dir "$RUN_DIR"
run_a100_paper_diagnostics "$RUN_DIR" "$EVENT_MANIFEST" "$SPLIT_MANIFEST" "$WINDOW_LENGTH" "$STRIDE"
"$PYTHON_BIN" -m neurotwin.cli report model-card --run-dir "$RUN_DIR"
