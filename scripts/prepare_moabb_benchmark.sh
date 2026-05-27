#!/usr/bin/env bash
set -euo pipefail

if (($#)); then
  OUT_DIR="$1"
elif [[ -n "${NEUROTWIN_DATA:-}" ]]; then
  OUT_DIR="$NEUROTWIN_DATA/prepared/moabb_benchmark"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "NEUROTWIN_DATA must be set or an output directory must be provided when running under SLURM." >&2
  exit 2
else
  OUT_DIR="/tmp/neurotwin_moabb_benchmark"
fi
MOABB_DATASET="${MOABB_DATASET:-BNCI2014_001}"
MOABB_PARADIGM="${MOABB_PARADIGM:-LeftRightImagery}"
WINDOW_LENGTH="${WINDOW_LENGTH:-128}"
STRIDE="${STRIDE:-128}"
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
  --out-dir "$OUT_DIR" \
  --require-windows

python -m neurotwin.cli eval \
  --suite neural_translation_v1 \
  --event-manifest "$OUT_DIR/event_manifest.json" \
  --split-manifest "$OUT_DIR/split_manifest.json" \
  --window-length "$WINDOW_LENGTH" \
  --stride "$STRIDE" \
  --train-steps "$TRAIN_STEPS" \
  --out-dir "$OUT_DIR"

python - "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
audit = json.loads((root / "eval_audit.json").read_text(encoding="utf-8"))
print(f"summary_event_count={audit.get('event_count')}")
print(f"summary_window_count={audit.get('window_count')}")
print(f"summary_window_counts_by_split={audit.get('window_counts_by_split')}")
suite_path = root / "prepared_baseline_suite.json"
if suite_path.exists():
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    tasks = suite.get("tasks", {})
    for task_id in sorted(tasks):
        payload = tasks.get(task_id, {})
        if isinstance(payload, dict):
            print(f"summary_task_status_{task_id}={payload.get('status')}")
PY
