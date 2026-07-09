#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
container_device_ids() {
  local count="$1"
  local ids=()
  local i
  for ((i = 0; i < count; i++)); do
    ids+=("$i")
  done
  local IFS=,
  echo "${ids[*]}"
}

export GPU_COUNT="${GPU_COUNT:-7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-$GPU_COUNT}"
export HOST_GPU_IDS="${HOST_GPU_IDS:-0,1,2,3,4,5,6}"
export CONTAINER_CUDA_VISIBLE_DEVICES="${CONTAINER_CUDA_VISIBLE_DEVICES:-$(container_device_ids "$GPU_COUNT")}"
export A100_CONFIG_TEMPLATE="${A100_CONFIG_TEMPLATE:-configs/train/moabb_a100.yaml}"
export A100_RUN_ID="${A100_RUN_ID:-moabb_a100}"

exec "$SCRIPT_DIR/run_docker_6gpu.sh" "$@"
