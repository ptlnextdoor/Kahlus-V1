#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/tmp/neurotwin_moabb_benchmark}"
MOABB_DATASET="${MOABB_DATASET:-BNCI2014_001}"
MOABB_PARADIGM="${MOABB_PARADIGM:-LeftRightImagery}"
WINDOW_LENGTH="${WINDOW_LENGTH:-1024}"
STRIDE="${STRIDE:-512}"
TRAIN_STEPS="${TRAIN_STEPS:-3}"
MAX_TRIALS="${MAX_TRIALS:-}"
SUBJECTS="${SUBJECTS:-}"

export PYTHONPATH="${PYTHONPATH:-}:src"

ARGS=(
  data prepare
  --dataset moabb
  --split subject
  --out-dir "$OUT_DIR"
  --moabb-dataset "$MOABB_DATASET"
  --moabb-paradigm "$MOABB_PARADIGM"
)
if [[ -n "$MAX_TRIALS" ]]; then
  ARGS+=(--max-trials "$MAX_TRIALS")
fi
if [[ -n "$SUBJECTS" ]]; then
  read -r -a SUBJECT_ARGS <<< "$SUBJECTS"
  ARGS+=(--subjects "${SUBJECT_ARGS[@]}")
fi

python -m neurotwin.cli "${ARGS[@]}"

python -m neurotwin.cli eval audit \
  --suite neural_translation_v1 \
  --event-manifest "$OUT_DIR/event_manifest.json" \
  --split-manifest "$OUT_DIR/split_manifest.json" \
  --window-length "$WINDOW_LENGTH" \
  --stride "$STRIDE" \
  --out-dir "$OUT_DIR"

python -m neurotwin.cli eval \
  --suite neural_translation_v1 \
  --event-manifest "$OUT_DIR/event_manifest.json" \
  --split-manifest "$OUT_DIR/split_manifest.json" \
  --window-length "$WINDOW_LENGTH" \
  --stride "$STRIDE" \
  --train-steps "$TRAIN_STEPS" \
  --out-dir "$OUT_DIR"
