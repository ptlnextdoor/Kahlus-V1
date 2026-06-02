#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PERSISTENT_ROOT:-}" || "$PERSISTENT_ROOT" != /* ]]; then
  echo "PERSISTENT_ROOT must be an absolute path inside the Docker container." >&2
  exit 2
fi
if [[ -z "${GPU_COUNT:-}" || ! "$GPU_COUNT" =~ ^[0-9]+$ ]] || ((GPU_COUNT < 1)); then
  echo "GPU_COUNT must be positive, got: ${GPU_COUNT:-}" >&2
  exit 2
fi
if [[ -z "${NPROC_PER_NODE:-}" || ! "$NPROC_PER_NODE" =~ ^[0-9]+$ ]] || ((NPROC_PER_NODE < 1)); then
  echo "NPROC_PER_NODE must be positive, got: ${NPROC_PER_NODE:-}" >&2
  exit 2
fi

cd /workspace/repo
export PYTHONPATH="${PYTHONPATH:-}:src"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
A100_CONFIG_TEMPLATE=${A100_CONFIG_TEMPLATE:-configs/train/moabb_a100_smoke.yaml}
A100_RUN_ID=${A100_RUN_ID:-$(basename "$A100_CONFIG_TEMPLATE" .yaml)}
A100_CONFIG_PATH=${A100_CONFIG_PATH:-outputs/configs/moabb_a100.materialized.yaml}
A100_REQUIRE_PAPER_MODE_GATE=${A100_REQUIRE_PAPER_MODE_GATE:-}
A100_PAPER_MODE_TRAIN_STEPS=${A100_PAPER_MODE_TRAIN_STEPS:-3}
if [[ -z "$A100_REQUIRE_PAPER_MODE_GATE" ]]; then
  A100_REQUIRE_PAPER_MODE_GATE=0
  if [[ "$A100_RUN_ID" != "moabb_a100_smoke" ]]; then
    A100_REQUIRE_PAPER_MODE_GATE=1
  fi
fi
if [[ ! -f "$A100_CONFIG_TEMPLATE" ]]; then
  echo "A100_CONFIG_TEMPLATE does not exist inside runner: $A100_CONFIG_TEMPLATE" >&2
  exit 2
fi
if [[ ! "$A100_RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "A100_RUN_ID must be a safe run directory name, got: $A100_RUN_ID" >&2
  exit 2
fi
MOABB_PREPARED_DIR="$PERSISTENT_ROOT/prepared/moabb_benchmark"
RUN_DIR="$PERSISTENT_ROOT/runs/$A100_RUN_ID"

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
  local prepared_dir=$2
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
  python -m neurotwin.cli eval leakage-demo \
    --seeds 0 1 2 \
    --event-manifest "$event_manifest" \
    --split-manifest "$split_manifest" \
    --window-length "$window_length" \
    --stride "$stride" \
    --out-dir "$run_dir"
  python -m neurotwin.cli eval identity-probe \
    --seeds 0 1 2 \
    --event-manifest "$event_manifest" \
    --split-manifest "$split_manifest" \
    --window-length "$window_length" \
    --stride "$stride" \
    --out-dir "$run_dir"
}

mkdir -p \
  "$PERSISTENT_ROOT/moabb" \
  "$PERSISTENT_ROOT/bids" \
  "$PERSISTENT_ROOT/prepared" \
  "$PERSISTENT_ROOT/runs" \
  "$PERSISTENT_ROOT/logs" \
  outputs/configs \
  outputs/smoke

python -m pip install -e ".[moabb,cluster]"
python scripts/docker_gpu_preflight.py "$PERSISTENT_ROOT/gpu_preflight.json"
bash scripts/run_smoke.sh outputs/smoke
bash scripts/prepare_moabb_benchmark.sh "$MOABB_PREPARED_DIR"
python -m neurotwin.cli eval audit \
  --suite neural_translation_v1 \
  --event-manifest "$MOABB_PREPARED_DIR/event_manifest.json" \
  --split-manifest "$MOABB_PREPARED_DIR/split_manifest.json" \
  --window-length 128 \
  --stride 128 \
  --out-dir "$MOABB_PREPARED_DIR" \
  --require-windows
python -m neurotwin.cli cluster materialize-config \
  --template "$A100_CONFIG_TEMPLATE" \
  --prepared-root "$MOABB_PREPARED_DIR" \
  --out "$A100_CONFIG_PATH"
python -m neurotwin.cli cluster preflight \
  --config "$A100_CONFIG_PATH" \
  --run-root "$PERSISTENT_ROOT/runs" \
  --require-cuda \
  --require-prepared-windows \
  --expect-window-count "${EXPECTED_WINDOW_COUNT:-18144}" \
  --expect-split-windows "${EXPECTED_SPLIT_WINDOWS:-train:12096,val:2016,test:4032}"
if [[ "$A100_REQUIRE_PAPER_MODE_GATE" == "1" ]]; then
  A100_PAPER_MODE_EVAL_DIR=${A100_PAPER_MODE_EVAL_DIR:-"$PERSISTENT_ROOT/eval/${A100_RUN_ID}_paper_mode"}
  python -m neurotwin.cli eval suite \
    --suite neural_translation_v1 \
    --paper-mode \
    --seeds 0 1 2 \
    --event-manifest "$MOABB_PREPARED_DIR/event_manifest.json" \
    --split-manifest "$MOABB_PREPARED_DIR/split_manifest.json" \
    --window-length 128 \
    --stride 128 \
    --train-steps "$A100_PAPER_MODE_TRAIN_STEPS" \
    --out-dir "$A100_PAPER_MODE_EVAL_DIR"
fi
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  -m neurotwin.cli train \
  --config "$A100_CONFIG_PATH" \
  --run-root "$PERSISTENT_ROOT/runs"
copy_a100_paper_artifacts "$RUN_DIR"
copy_prepared_eval_audit "$RUN_DIR" "$MOABB_PREPARED_DIR"
python -m neurotwin.cli report --run-dir "$RUN_DIR"
run_a100_paper_diagnostics "$RUN_DIR" "$MOABB_PREPARED_DIR/event_manifest.json" "$MOABB_PREPARED_DIR/split_manifest.json" 128 128
python -m neurotwin.cli report model-card --run-dir "$RUN_DIR"
bash scripts/package_a100_evidence_bundle.sh "$PERSISTENT_ROOT" outputs
