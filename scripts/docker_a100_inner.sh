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
  --template configs/train/moabb_a100_smoke.yaml \
  --prepared-root "$PERSISTENT_ROOT/prepared/moabb_benchmark" \
  --out outputs/configs/moabb_a100.materialized.yaml
python -m neurotwin.cli cluster preflight \
  --config outputs/configs/moabb_a100.materialized.yaml \
  --run-root "$PERSISTENT_ROOT/runs" \
  --require-cuda \
  --require-prepared-windows \
  --expect-window-count "${EXPECTED_WINDOW_COUNT:-18144}" \
  --expect-split-windows "${EXPECTED_SPLIT_WINDOWS:-train:12096,val:2016,test:4032}"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  -m neurotwin.cli train \
  --config outputs/configs/moabb_a100.materialized.yaml \
  --run-root "$PERSISTENT_ROOT/runs"
python -m neurotwin.cli report --run-dir "$PERSISTENT_ROOT/runs/moabb_a100_smoke"
bash scripts/package_a100_evidence_bundle.sh "$PERSISTENT_ROOT" outputs
