#!/usr/bin/env bash
# Host-side Docker launcher for the Kahlus v3 KTM synthetic A100 micro-sweep (SYNTHETIC ONLY).
# Mounts the runner at /workspace/repo, exposes GPU_COUNT GPUs, and runs docker_ktm_inner.sh.
# Supports GPU_COUNT=1 (diagnostic), 6, or 8. NO MOABB / real-data env. DO NOT run on A100 until
# local CPU smoke + checksum verification pass.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: scripts/run_docker_ktm.sh /raid/scratch/$USER/kahlus-ktm-<short_sha> [host-gpu-ids]

Examples (set GPU_COUNT to 1, 6, or 8):
  GPU_COUNT=8 HOST_GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/run_docker_ktm.sh /raid/scratch/$USER/kahlus-ktm-abc1234
  GPU_COUNT=6 HOST_GPU_IDS=0,1,2,3,4,5     bash scripts/run_docker_ktm.sh /raid/scratch/$USER/kahlus-ktm-abc1234
  GPU_COUNT=1 HOST_GPU_IDS=0               bash scripts/run_docker_ktm.sh /raid/scratch/$USER/kahlus-ktm-abc1234

Exposing GPUs to the container is NOT enough: torchrun --nproc_per_node must equal GPU_COUNT,
which docker_ktm_inner.sh enforces via NPROC_PER_NODE.
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
GPU_COUNT=${GPU_COUNT:-8}
HOST_GPU_IDS=${HOST_GPU_IDS:-${2:-0,1,2,3,4,5,6,7}}
NPROC_PER_NODE=${NPROC_PER_NODE:-$GPU_COUNT}
KTM_CONFIG=${KTM_CONFIG:-configs/train/ktm_a100_micro.yaml}
NEUROTWIN_DOCKER_DRY_RUN=${NEUROTWIN_DOCKER_DRY_RUN:-0}

for name in GPU_COUNT NPROC_PER_NODE; do
  value=${!name}
  if ! [[ "$value" =~ ^[0-9]+$ ]] || ((value < 1)); then
    echo "$name must be positive, got: $value" >&2
    exit 2
  fi
done

if [[ -z "${CONTAINER_CUDA_VISIBLE_DEVICES:-}" ]]; then
  CONTAINER_CUDA_VISIBLE_DEVICES=0
  for ((index = 1; index < GPU_COUNT; index++)); do
    CONTAINER_CUDA_VISIBLE_DEVICES+=",$index"
  done
fi

if [[ "$INPUT_ROOT" != /* ]]; then
  echo "Persistent root must be absolute, got: $INPUT_ROOT" >&2
  exit 2
fi
if [[ "${NEUROTWIN_ALLOW_LOCAL_PERSISTENT_ROOT:-0}" != "1" ]]; then
  case "$INPUT_ROOT" in
    /tmp|/tmp/*|/private/tmp|/private/tmp/*|/var/tmp|/var/tmp/*)
      echo "Persistent root must not be local tmp: $INPUT_ROOT" >&2; exit 2 ;;
    /Users|/Users/*)
      echo "Persistent root must be on cluster scratch, not a local laptop path: $INPUT_ROOT" >&2; exit 2 ;;
    "$REPO_ROOT"|"$REPO_ROOT"/*)
      echo "Persistent root must not be inside the extracted runner: $INPUT_ROOT" >&2; exit 2 ;;
  esac
fi

mkdir -p "$INPUT_ROOT"
PERSISTENT_ROOT="$(cd "$INPUT_ROOT" && pwd)"
RUN_LOG_DIR="$PERSISTENT_ROOT/logs"
mkdir -p "$RUN_LOG_DIR"
DOCKER_RUN_ID=${DOCKER_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
DOCKER_LOG_PATH="$RUN_LOG_DIR/kahlus-ktm-docker-$DOCKER_RUN_ID.log"

{
  printf 'DOCKER_RUN_ID=%s\n' "$DOCKER_RUN_ID"
  printf 'DOCKER_LOG_PATH=%s\n' "$DOCKER_LOG_PATH"
  printf 'DOCKER_IMAGE=%s\n' "$DOCKER_IMAGE"
  printf 'HOST_GPU_IDS=%s\n' "$HOST_GPU_IDS"
  printf 'GPU_COUNT=%s\n' "$GPU_COUNT"
  printf 'NPROC_PER_NODE=%s\n' "$NPROC_PER_NODE"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CONTAINER_CUDA_VISIBLE_DEVICES"
  printf 'KTM_CONFIG=%s\n' "$KTM_CONFIG"
} > "$PERSISTENT_ROOT/docker_run.env"

echo "docker_image=$DOCKER_IMAGE persistent_root=$PERSISTENT_ROOT host_gpu_ids=$HOST_GPU_IDS"
echo "gpu_count=$GPU_COUNT nproc_per_node=$NPROC_PER_NODE cuda_visible_devices=$CONTAINER_CUDA_VISIBLE_DEVICES"

if [[ "$NEUROTWIN_DOCKER_DRY_RUN" == "1" ]]; then
  echo "docker_dry_run=true"
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required for scripts/run_docker_ktm.sh." >&2
  exit 2
fi

docker run --rm -i --gpus "\"device=${HOST_GPU_IDS}\"" \
  --ipc=host \
  --shm-size=64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$REPO_ROOT":/workspace/repo \
  -v "$PERSISTENT_ROOT":"$PERSISTENT_ROOT" \
  -w /workspace/repo \
  -e PERSISTENT_ROOT="$PERSISTENT_ROOT" \
  -e GPU_COUNT="$GPU_COUNT" \
  -e HOST_GPU_IDS="$HOST_GPU_IDS" \
  -e CUDA_VISIBLE_DEVICES="$CONTAINER_CUDA_VISIBLE_DEVICES" \
  -e NPROC_PER_NODE="$NPROC_PER_NODE" \
  -e KTM_CONFIG="$KTM_CONFIG" \
  -e NCCL_DEBUG="${NCCL_DEBUG:-INFO}" \
  -e DOCKER_IMAGE="$DOCKER_IMAGE" \
  -e DOCKER_LOG_PATH="$DOCKER_LOG_PATH" \
  "$DOCKER_IMAGE" \
  bash scripts/docker_ktm_inner.sh 2>&1 | tee -a "$DOCKER_LOG_PATH"
