#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: scripts/run_full.sh /shared/persistent/neurotwin" >&2
}

if (($# != 1)); then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT_ROOT=$1
if [[ "$INPUT_ROOT" != /* ]]; then
  echo "Persistent root must be absolute, got: $INPUT_ROOT" >&2
  exit 2
fi
case "$INPUT_ROOT" in
  /path/to|/path/to/*|/absolute|/absolute/*)
    echo "Persistent root must be a real Chapman shared path, not placeholder text: $INPUT_ROOT" >&2
    exit 2
    ;;
  /tmp|/tmp/*|/private/tmp|/private/tmp/*|/var/tmp|/var/tmp/*)
    echo "Persistent root must not be local tmp: $INPUT_ROOT" >&2
    exit 2
    ;;
  /Users|/Users/*)
    echo "Persistent root must be on a Chapman shared filesystem, not a local laptop path: $INPUT_ROOT" >&2
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

cd "$REPO_ROOT"

PERSISTENT_ROOT="$(cd "$INPUT_ROOT" && pwd)"
case "$PERSISTENT_ROOT" in
  "$REPO_ROOT"|"$REPO_ROOT"/*)
    echo "Persistent root must not be inside the checkout: $PERSISTENT_ROOT" >&2
    exit 2
    ;;
esac
export NEUROTWIN_DATA="$PERSISTENT_ROOT"
export MOABB_DATA="$NEUROTWIN_DATA/moabb"
export BIDS_ROOT="$NEUROTWIN_DATA/bids"
export RUN_ROOT="$NEUROTWIN_DATA/runs"
export RUN_LOG_DIR="$NEUROTWIN_DATA/logs"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${PYTHONPATH:-}:src"

mkdir -p logs outputs/configs "$MOABB_DATA" "$BIDS_ROOT" "$RUN_ROOT" "$RUN_LOG_DIR" "$NEUROTWIN_DATA/prepared"
if [[ ! -w logs ]]; then
  echo "logs/ is not writable." >&2
  exit 2
fi
if [[ ! -w "$RUN_LOG_DIR" ]]; then
  echo "Persistent log directory is not writable: $RUN_LOG_DIR" >&2
  exit 2
fi

PREPARED_DIR="$NEUROTWIN_DATA/prepared/moabb_benchmark"
EVENT_MANIFEST="$PREPARED_DIR/event_manifest.json"
SPLIT_MANIFEST="$PREPARED_DIR/split_manifest.json"
EXPECTED_WINDOW_COUNT="${EXPECTED_WINDOW_COUNT:-18144}"
EXPECTED_TRAIN_WINDOWS="${EXPECTED_TRAIN_WINDOWS:-12096}"
EXPECTED_VAL_WINDOWS="${EXPECTED_VAL_WINDOWS:-2016}"
EXPECTED_TEST_WINDOWS="${EXPECTED_TEST_WINDOWS:-4032}"
EXPECTED_SPLIT_WINDOWS="${EXPECTED_SPLIT_WINDOWS:-train:$EXPECTED_TRAIN_WINDOWS,val:$EXPECTED_VAL_WINDOWS,test:$EXPECTED_TEST_WINDOWS}"

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

CONFIG_PATH="outputs/configs/moabb_a100.materialized.yaml"
echo "step=materialize_config path=$CONFIG_PATH"
"$PYTHON_BIN" -m neurotwin.cli cluster materialize-config \
  --template configs/train/moabb_a100_smoke.yaml \
  --prepared-root "$PREPARED_DIR" \
  --out "$CONFIG_PATH"

echo "step=login_node_preflight_without_cuda"
"$PYTHON_BIN" -m neurotwin.cli cluster preflight \
  --config "$CONFIG_PATH" \
  --run-root "$RUN_ROOT" \
  --require-prepared-windows \
  --expect-window-count "$EXPECTED_WINDOW_COUNT" \
  --expect-split-windows "$EXPECTED_SPLIT_WINDOWS"

SBATCH_ARGS=()
if [[ -n "${SBATCH_PARTITION:-}" ]]; then
  SBATCH_ARGS+=(--partition "$SBATCH_PARTITION")
fi
if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then
  SBATCH_ARGS+=(--account "$SBATCH_ACCOUNT")
fi
if [[ -n "${SBATCH_QOS:-}" ]]; then
  SBATCH_ARGS+=(--qos "$SBATCH_QOS")
fi

echo "step=submit_a100_job"
export REPO_ROOT EXPECTED_WINDOW_COUNT EXPECTED_SPLIT_WINDOWS RUN_LOG_DIR
sbatch "${SBATCH_ARGS[@]}" \
  --output "$RUN_LOG_DIR/neurotwin-a100-full-%j.out" \
  --error "$RUN_LOG_DIR/neurotwin-a100-full-%j.err" \
  --export=ALL \
  scripts/run_full.sbatch "$CONFIG_PATH" "$RUN_ROOT"
