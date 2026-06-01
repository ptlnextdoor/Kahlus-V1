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
bash scripts/prepare_moabb_benchmark.sh "$PERSISTENT_ROOT/prepared/moabb_benchmark"
python -m neurotwin.cli eval audit \
  --suite neural_translation_v1 \
  --event-manifest "$PERSISTENT_ROOT/prepared/moabb_benchmark/event_manifest.json" \
  --split-manifest "$PERSISTENT_ROOT/prepared/moabb_benchmark/split_manifest.json" \
  --window-length 128 \
  --stride 128 \
  --out-dir "$PERSISTENT_ROOT/prepared/moabb_benchmark" \
  --require-windows
python -m neurotwin.cli cluster materialize-config \
  --template "$A100_CONFIG_TEMPLATE" \
  --prepared-root "$PERSISTENT_ROOT/prepared/moabb_benchmark" \
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
  python -m neurotwin.cli eval \
    --suite neural_translation_v1 \
    --paper-mode \
    --seeds 0 1 2 \
    --event-manifest "$PERSISTENT_ROOT/prepared/moabb_benchmark/event_manifest.json" \
    --split-manifest "$PERSISTENT_ROOT/prepared/moabb_benchmark/split_manifest.json" \
    --window-length 128 \
    --stride 128 \
    --train-steps "$A100_PAPER_MODE_TRAIN_STEPS" \
    --out-dir "$A100_PAPER_MODE_EVAL_DIR"
fi
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  -m neurotwin.cli train \
  --config "$A100_CONFIG_PATH" \
  --run-root "$PERSISTENT_ROOT/runs"
if [[ -n "${A100_PAPER_MODE_EVAL_DIR:-}" && -f "$A100_PAPER_MODE_EVAL_DIR/paper_mode_gate.json" ]]; then
  cp "$A100_PAPER_MODE_EVAL_DIR/paper_mode_gate.json" "$PERSISTENT_ROOT/runs/$A100_RUN_ID/paper_mode_gate.json"
fi
python -m neurotwin.cli report --run-dir "$PERSISTENT_ROOT/runs/$A100_RUN_ID"
bash scripts/package_a100_evidence_bundle.sh "$PERSISTENT_ROOT" outputs
