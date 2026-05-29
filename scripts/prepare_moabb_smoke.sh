#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/moabb_prepare_common.sh"

OUT_DIR="$(resolve_moabb_out_dir moabb_smoke "$@")"
MOABB_DATASET="${MOABB_DATASET:-BNCI2014_001}"
MOABB_PARADIGM="${MOABB_PARADIGM:-LeftRightImagery}"
MAX_TRIALS="${MAX_TRIALS:-12}"
WINDOW_LENGTH="${WINDOW_LENGTH:-128}"
STRIDE="${STRIDE:-128}"
TRAIN_STEPS="${TRAIN_STEPS:-1}"
SUBJECTS="${SUBJECTS:-1 2 3}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="${PYTHONPATH:-}:src"

read -r -a SUBJECT_ARGS <<< "$SUBJECTS"

"$PYTHON_BIN" -m neurotwin.cli data smoke \
  --dataset moabb \
  --split subject \
  --out-dir "$OUT_DIR" \
  --moabb-dataset "$MOABB_DATASET" \
  --moabb-paradigm "$MOABB_PARADIGM" \
  --subjects "${SUBJECT_ARGS[@]}" \
  --max-trials "$MAX_TRIALS" \
  --window-length "$WINDOW_LENGTH" \
  --stride "$STRIDE" \
  --train-steps "$TRAIN_STEPS" \
  --require-windows

"$PYTHON_BIN" -m neurotwin.cli eval audit \
  --suite neural_translation_v1 \
  --event-manifest "$OUT_DIR/event_manifest.json" \
  --split-manifest "$OUT_DIR/split_manifest.json" \
  --window-length "$WINDOW_LENGTH" \
  --stride "$STRIDE" \
  --out-dir "$OUT_DIR" \
  --require-windows

print_moabb_prepare_summary "$OUT_DIR"
