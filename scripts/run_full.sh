#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: scripts/run_full.sh /shared/persistent/neurotwin" >&2
}

if (($# != 1)); then
  usage
  exit 2
fi

INPUT_ROOT=$1
if [[ "$INPUT_ROOT" != /* ]]; then
  echo "Persistent root must be absolute, got: $INPUT_ROOT" >&2
  exit 2
fi
case "$INPUT_ROOT" in
  /tmp|/tmp/*|/private/tmp|/private/tmp/*|/var/tmp|/var/tmp/*)
    echo "Persistent root must not be local tmp: $INPUT_ROOT" >&2
    exit 2
    ;;
esac
if [[ ! -d "$INPUT_ROOT" ]]; then
  echo "Persistent root does not exist. Create it first: $INPUT_ROOT" >&2
  exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is required for the full A100 run." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PERSISTENT_ROOT="$(cd "$INPUT_ROOT" && pwd)"
export NEUROTWIN_DATA="$PERSISTENT_ROOT"
export MOABB_DATA="$NEUROTWIN_DATA/moabb"
export BIDS_ROOT="$NEUROTWIN_DATA/bids"
export RUN_ROOT="$NEUROTWIN_DATA/runs"
export PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p logs outputs/configs "$MOABB_DATA" "$BIDS_ROOT" "$RUN_ROOT" "$NEUROTWIN_DATA/prepared"
if [[ ! -w logs ]]; then
  echo "logs/ is not writable." >&2
  exit 2
fi

PREPARED_DIR="$NEUROTWIN_DATA/prepared/moabb_benchmark"
EVENT_MANIFEST="$PREPARED_DIR/event_manifest.json"
SPLIT_MANIFEST="$PREPARED_DIR/split_manifest.json"
EXPECTED_WINDOW_COUNT="${EXPECTED_WINDOW_COUNT:-18144}"
EXPECTED_TRAIN_WINDOWS="${EXPECTED_TRAIN_WINDOWS:-12096}"
EXPECTED_VAL_WINDOWS="${EXPECTED_VAL_WINDOWS:-2016}"
EXPECTED_TEST_WINDOWS="${EXPECTED_TEST_WINDOWS:-4032}"

if [[ ! -f "$EVENT_MANIFEST" || ! -f "$SPLIT_MANIFEST" ]]; then
  echo "step=prepare_moabb_benchmark"
  bash scripts/prepare_moabb_benchmark.sh "$PREPARED_DIR"
fi
if [[ ! -f "$EVENT_MANIFEST" || ! -f "$SPLIT_MANIFEST" ]]; then
  echo "MOABB preparation did not write required manifests under $PREPARED_DIR" >&2
  exit 2
fi

echo "step=refresh_eval_audit"
"$PYTHON_BIN" -m neurotwin.cli eval audit \
  --suite neural_translation_v1 \
  --event-manifest "$EVENT_MANIFEST" \
  --split-manifest "$SPLIT_MANIFEST" \
  --window-length 128 \
  --stride 128 \
  --out-dir "$PREPARED_DIR" \
  --require-windows

echo "step=verify_exact_window_gate"
"$PYTHON_BIN" - "$PREPARED_DIR/eval_audit.json" "$EXPECTED_WINDOW_COUNT" "$EXPECTED_TRAIN_WINDOWS" "$EXPECTED_VAL_WINDOWS" "$EXPECTED_TEST_WINDOWS" <<'PY'
import json
import sys
from pathlib import Path

audit = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_total = int(sys.argv[2])
expected = {"train": int(sys.argv[3]), "val": int(sys.argv[4]), "test": int(sys.argv[5])}
counts = {key: int(audit.get("window_counts_by_split", {}).get(key, 0)) for key in expected}
if not audit.get("passed"):
    raise SystemExit(f"eval audit failed: {audit.get('violations')}")
if int(audit.get("window_count", 0)) != expected_total:
    raise SystemExit(f"expected window_count={expected_total}, got {audit.get('window_count')}")
if counts != expected:
    raise SystemExit(f"expected split windows={expected}, got {counts}")
print("eval_audit_passed=True")
print(f"window_count={expected_total}")
print("window_counts_by_split=" + ",".join(f"{key}:{counts[key]}" for key in ("train", "val", "test")))
PY

CONFIG_PATH="outputs/configs/moabb_a100.materialized.yaml"
echo "step=materialize_config path=$CONFIG_PATH"
"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

template = Path("configs/train/moabb_a100_smoke.yaml")
out = Path("outputs/configs/moabb_a100.materialized.yaml")
root = Path(os.environ["NEUROTWIN_DATA"]).resolve()
text = template.read_text(encoding="utf-8")
text = text.replace(
    "event_manifest: /path/to/moabb_prepared/event_manifest.json",
    f"event_manifest: {root}/prepared/moabb_benchmark/event_manifest.json",
)
text = text.replace(
    "split_manifest: /path/to/moabb_prepared/split_manifest.json",
    f"split_manifest: {root}/prepared/moabb_benchmark/split_manifest.json",
)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(text, encoding="utf-8")
PY

echo "step=login_node_preflight_without_cuda"
"$PYTHON_BIN" -m neurotwin.cli cluster preflight \
  --config "$CONFIG_PATH" \
  --run-root "$RUN_ROOT" \
  --require-prepared-windows

SBATCH_ARGS=()
if [[ -n "${SBATCH_PARTITION:-}" ]]; then
  SBATCH_ARGS+=(--partition "$SBATCH_PARTITION")
fi
if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then
  SBATCH_ARGS+=(--account "$SBATCH_ACCOUNT")
fi

echo "step=submit_a100_job"
sbatch "${SBATCH_ARGS[@]}" \
  --export=ALL,NEUROTWIN_DATA="$NEUROTWIN_DATA",MOABB_DATA="$MOABB_DATA",BIDS_ROOT="$BIDS_ROOT",RUN_ROOT="$RUN_ROOT",PYTHON_BIN="$PYTHON_BIN" \
  scripts/run_full.sbatch "$CONFIG_PATH" "$RUN_ROOT"
