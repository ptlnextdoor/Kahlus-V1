#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/tmp/neurotwin_moabb_smoke}"
MOABB_DATASET="${MOABB_DATASET:-BNCI2014_001}"
MOABB_PARADIGM="${MOABB_PARADIGM:-LeftRightImagery}"
MAX_TRIALS="${MAX_TRIALS:-12}"
WINDOW_LENGTH="${WINDOW_LENGTH:-128}"
STRIDE="${STRIDE:-128}"
TRAIN_STEPS="${TRAIN_STEPS:-1}"
SUBJECTS="${SUBJECTS:-1 2 3}"

export PYTHONPATH="${PYTHONPATH:-}:src"

python -m neurotwin.cli data smoke \
  --dataset moabb \
  --split subject \
  --out-dir "$OUT_DIR" \
  --moabb-dataset "$MOABB_DATASET" \
  --moabb-paradigm "$MOABB_PARADIGM" \
  --subjects $SUBJECTS \
  --max-trials "$MAX_TRIALS" \
  --window-length "$WINDOW_LENGTH" \
  --stride "$STRIDE" \
  --train-steps "$TRAIN_STEPS"

python -m neurotwin.cli eval audit \
  --suite neural_translation_v1 \
  --event-manifest "$OUT_DIR/event_manifest.json" \
  --split-manifest "$OUT_DIR/split_manifest.json" \
  --window-length "$WINDOW_LENGTH" \
  --stride "$STRIDE" \
  --out-dir "$OUT_DIR"
