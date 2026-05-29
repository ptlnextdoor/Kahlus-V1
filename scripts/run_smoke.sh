#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-outputs/smoke}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "$OUT_DIR/prepared" "$OUT_DIR/configs" "$OUT_DIR/runs"
export PYTHONPATH="${PYTHONPATH:-}:src"

echo "step=doctor"
"$PYTHON_BIN" -m neurotwin.cli doctor

echo "step=prepare_synthetic"
"$PYTHON_BIN" -m neurotwin.cli data prepare \
  --dataset synthetic \
  --split subject \
  --out-dir "$OUT_DIR/prepared"

CONFIG_PATH="$OUT_DIR/configs/prepared_synthetic_debug.yaml"
echo "step=materialize_smoke_config path=$CONFIG_PATH"
"$PYTHON_BIN" - "$CONFIG_PATH" "$OUT_DIR/prepared" <<'PY'
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
prepared = Path(sys.argv[2]).resolve()
text = Path("configs/train/prepared_synthetic_debug.yaml").read_text(encoding="utf-8")
text = text.replace("/tmp/neurotwin_prepared/event_manifest.json", str(prepared / "event_manifest.json"))
text = text.replace("/tmp/neurotwin_prepared/split_manifest.json", str(prepared / "split_manifest.json"))
config_path.write_text(text, encoding="utf-8")
PY

echo "step=train_smoke"
"$PYTHON_BIN" -m neurotwin.cli train \
  --config "$CONFIG_PATH" \
  --run-root "$OUT_DIR/runs"

RUN_DIR="$OUT_DIR/runs/prepared_synthetic_debug"
echo "step=report_smoke"
"$PYTHON_BIN" -m neurotwin.cli report --run-dir "$RUN_DIR"

echo "smoke_status=completed"
echo "smoke_run_dir=$RUN_DIR"
