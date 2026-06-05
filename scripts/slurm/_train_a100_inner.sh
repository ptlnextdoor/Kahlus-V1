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
A100_RUN_PAPER_MODE_IN_FULL="${A100_RUN_PAPER_MODE_IN_FULL:-0}"
A100_PAPER_MODE_TRAIN_STEPS="${A100_PAPER_MODE_TRAIN_STEPS:-3}"
A100_PAPER_MODE_EVAL_DIR="${A100_PAPER_MODE_EVAL_DIR:-}"

if [[ "$CONFIG" != /* ]]; then
  echo "Config must be a materialized absolute path before A100 launch, got: $CONFIG" >&2
  exit 2
fi
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

paper_mode_gate_passed() {
  local eval_dir=$1
  "$PYTHON_BIN" - "$eval_dir/paper_mode_gate.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except json.JSONDecodeError:
    raise SystemExit(1)
raise SystemExit(0 if payload.get("passed") is True else 1)
PY
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

if [[ "$A100_RUN_PAPER_MODE_IN_FULL" == "1" ]]; then
  A100_PAPER_MODE_EVAL_DIR=${A100_PAPER_MODE_EVAL_DIR:-"$(dirname "$RUN_ROOT")/eval/$(basename "$RUN_DIR")_paper_mode"}
  "$PYTHON_BIN" -m neurotwin.cli eval suite \
    --suite neural_translation_v1 \
    --paper-mode \
    --seeds 0 1 2 \
    --event-manifest "$EVENT_MANIFEST" \
    --split-manifest "$SPLIT_MANIFEST" \
    --window-length "$WINDOW_LENGTH" \
    --stride "$STRIDE" \
    --train-steps "$A100_PAPER_MODE_TRAIN_STEPS" \
    --out-dir "$A100_PAPER_MODE_EVAL_DIR"
elif [[ -n "$A100_PAPER_MODE_EVAL_DIR" ]] && paper_mode_gate_passed "$A100_PAPER_MODE_EVAL_DIR"; then
  echo "paper_mode_phase1_artifacts=$A100_PAPER_MODE_EVAL_DIR"
else
  echo "paper_mode_artifacts_unavailable=Phase 1 artifacts missing; set A100_RUN_PAPER_MODE_IN_FULL=1 to run inside full allocation." >&2
fi

torchrun --standalone --nproc_per_node="$NPROC" \
  -m neurotwin.cli train --config "$CONFIG" --run-root "$RUN_ROOT"

"$PYTHON_BIN" -m neurotwin.cli run finalize \
  --run-dir "$RUN_DIR" \
  --paper-mode-dir "${A100_PAPER_MODE_EVAL_DIR:-}" \
  --event-manifest "$EVENT_MANIFEST" \
  --split-manifest "$SPLIT_MANIFEST" \
  --window-length "$WINDOW_LENGTH" \
  --stride "$STRIDE" \
  --seeds 0 1 2
