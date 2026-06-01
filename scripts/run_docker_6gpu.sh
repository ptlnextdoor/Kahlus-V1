#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: scripts/run_docker_6gpu.sh /raid/scratch/$USER/neurotwin-<short_sha> [host-gpu-ids]

Examples:
  bash scripts/run_docker_6gpu.sh /raid/scratch/$USER/neurotwin-abc1234
  HOST_GPU_IDS=2,3,4,5,6,7 bash scripts/run_docker_6gpu.sh /raid/scratch/$USER/neurotwin-abc1234
  GPU_COUNT=1 NPROC_PER_NODE=1 HOST_GPU_IDS=0 CUDA_VISIBLE_DEVICES=0 bash scripts/run_docker_6gpu.sh /raid/scratch/$USER/neurotwin-abc1234

Runs the A100 handoff inside Docker with the runner mounted at /workspace/repo.
The default path requires exactly 6 visible CUDA devices. Use the one-GPU form only for diagnostics.
EOF
}

if (($# < 1 || $# > 2)); then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INPUT_ROOT=$1
DOCKER_IMAGE=${DOCKER_IMAGE:-pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel}
GPU_COUNT=${GPU_COUNT:-6}
HOST_GPU_IDS=${HOST_GPU_IDS:-${2:-0,1,2,3,4,5}}
NPROC_PER_NODE=${NPROC_PER_NODE:-$GPU_COUNT}
for name in GPU_COUNT NPROC_PER_NODE; do
  value=${!name}
  if ! [[ "$value" =~ ^[0-9]+$ ]] || ((value < 1)); then
    echo "$name must be positive, got: $value" >&2
    exit 2
  fi
done
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CONTAINER_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
else
  CONTAINER_CUDA_VISIBLE_DEVICES=0
  for ((index = 1; index < GPU_COUNT; index++)); do
    CONTAINER_CUDA_VISIBLE_DEVICES+=",$index"
  done
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required for scripts/run_docker_6gpu.sh." >&2
  exit 2
fi
if [[ "$INPUT_ROOT" != /* ]]; then
  echo "Persistent root must be absolute, got: $INPUT_ROOT" >&2
  exit 2
fi
case "$INPUT_ROOT" in
  /tmp|/tmp/*|/private/tmp|/private/tmp/*|/var/tmp|/var/tmp/*)
    echo "Persistent root must not be local tmp: $INPUT_ROOT" >&2
    exit 2
    ;;
  /Users|/Users/*)
    echo "Persistent root must be on the cluster scratch/shared filesystem, not a local laptop path: $INPUT_ROOT" >&2
    exit 2
    ;;
  "$REPO_ROOT"|"$REPO_ROOT"/*)
    echo "Persistent root must not be inside the extracted runner: $INPUT_ROOT" >&2
    exit 2
    ;;
esac

mkdir -p "$INPUT_ROOT"
PERSISTENT_ROOT="$(cd "$INPUT_ROOT" && pwd)"
DOCKER_TTY=(-i)
if [[ -t 0 && -t 1 ]]; then
  DOCKER_TTY=(-it)
fi
RUN_LOG_DIR="$PERSISTENT_ROOT/logs"
mkdir -p "$RUN_LOG_DIR"
DOCKER_RUN_ID=${DOCKER_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
DOCKER_LOG_PATH=${DOCKER_LOG_PATH:-"$RUN_LOG_DIR/neurotwin-a100-docker-$DOCKER_RUN_ID.log"}
if [[ "$(dirname "$DOCKER_LOG_PATH")" != "$RUN_LOG_DIR" || ! "$(basename "$DOCKER_LOG_PATH")" =~ ^neurotwin-a100-docker-[A-Za-z0-9_.:-]+\.log$ ]]; then
  echo "DOCKER_LOG_PATH must be under $RUN_LOG_DIR and named neurotwin-a100-docker-*.log, got: $DOCKER_LOG_PATH" >&2
  exit 2
fi
{
  printf 'DOCKER_RUN_ID=%s\n' "$DOCKER_RUN_ID"
  printf 'DOCKER_LOG_PATH=%s\n' "$DOCKER_LOG_PATH"
  printf 'DOCKER_IMAGE=%s\n' "$DOCKER_IMAGE"
  printf 'HOST_GPU_IDS=%s\n' "$HOST_GPU_IDS"
  printf 'GPU_COUNT=%s\n' "$GPU_COUNT"
  printf 'NPROC_PER_NODE=%s\n' "$NPROC_PER_NODE"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CONTAINER_CUDA_VISIBLE_DEVICES"
} > "$PERSISTENT_ROOT/docker_run.env"
exec > >(tee -a "$DOCKER_LOG_PATH") 2>&1

echo "docker_image=$DOCKER_IMAGE"
echo "persistent_root=$PERSISTENT_ROOT"
echo "host_gpu_ids=$HOST_GPU_IDS"
echo "cuda_visible_devices=$CONTAINER_CUDA_VISIBLE_DEVICES"
echo "gpu_count=$GPU_COUNT"
echo "nproc_per_node=$NPROC_PER_NODE"
echo "docker_log_path=$DOCKER_LOG_PATH"

docker run --rm "${DOCKER_TTY[@]}" --gpus "\"device=${HOST_GPU_IDS}\"" \
  --ipc=host \
  --shm-size=64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$REPO_ROOT":/workspace/repo \
  -v "$PERSISTENT_ROOT":"$PERSISTENT_ROOT" \
  -w /workspace/repo \
  -e PERSISTENT_ROOT="$PERSISTENT_ROOT" \
  -e NEUROTWIN_DATA="$PERSISTENT_ROOT" \
  -e MOABB_DATA="$PERSISTENT_ROOT/moabb" \
  -e BIDS_ROOT="$PERSISTENT_ROOT/bids" \
  -e RUN_ROOT="$PERSISTENT_ROOT/runs" \
  -e EXPECTED_WINDOW_COUNT="${EXPECTED_WINDOW_COUNT:-18144}" \
  -e EXPECTED_SPLIT_WINDOWS="${EXPECTED_SPLIT_WINDOWS:-train:12096,val:2016,test:4032}" \
  -e GPU_COUNT="$GPU_COUNT" \
  -e HOST_GPU_IDS="$HOST_GPU_IDS" \
  -e CUDA_VISIBLE_DEVICES="$CONTAINER_CUDA_VISIBLE_DEVICES" \
  -e NPROC_PER_NODE="$NPROC_PER_NODE" \
  -e NCCL_DEBUG="${NCCL_DEBUG:-INFO}" \
  -e DOCKER_IMAGE="$DOCKER_IMAGE" \
  -e DOCKER_LOG_PATH="$DOCKER_LOG_PATH" \
  "$DOCKER_IMAGE" \
  bash scripts/docker_a100_inner.sh
