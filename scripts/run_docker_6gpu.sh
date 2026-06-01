#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: scripts/run_docker_6gpu.sh /raid/scratch/$USER/neurotwin-<short_sha> [gpu-list]

Example:
  bash scripts/run_docker_6gpu.sh /raid/scratch/$USER/neurotwin-abc1234 0,1,2,3,4,5

Runs the A100 handoff inside Docker with the runner mounted at /workspace/repo.
EOF
}

if (($# < 1 || $# > 2)); then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INPUT_ROOT=$1
GPU_LIST=${2:-0,1,2,3,4,5}
DOCKER_IMAGE=${DOCKER_IMAGE:-pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel}

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

IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"
NPROC_PER_NODE=${NPROC_PER_NODE:-${#GPU_IDS[@]}}
if ! [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ ]] || ((NPROC_PER_NODE < 1)); then
  echo "NPROC_PER_NODE must be positive, got: $NPROC_PER_NODE" >&2
  exit 2
fi

mkdir -p "$INPUT_ROOT"
PERSISTENT_ROOT="$(cd "$INPUT_ROOT" && pwd)"

DOCKER_TTY=(-i)
if [[ -t 0 && -t 1 ]]; then
  DOCKER_TTY=(-it)
fi

echo "docker_image=$DOCKER_IMAGE"
echo "persistent_root=$PERSISTENT_ROOT"
echo "gpu_list=$GPU_LIST"
echo "nproc_per_node=$NPROC_PER_NODE"

docker run --rm "${DOCKER_TTY[@]}" --gpus "device=$GPU_LIST" \
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
  -e NPROC_PER_NODE="$NPROC_PER_NODE" \
  "$DOCKER_IMAGE" \
  bash -lc '
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$PERSISTENT_ROOT/moabb" "$PERSISTENT_ROOT/bids" "$PERSISTENT_ROOT/prepared" "$PERSISTENT_ROOT/runs" "$PERSISTENT_ROOT/logs" outputs/configs outputs/smoke

python -m pip install -e ".[moabb,cluster]"
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
  --expect-window-count "$EXPECTED_WINDOW_COUNT" \
  --expect-split-windows "$EXPECTED_SPLIT_WINDOWS"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  -m neurotwin.cli train \
  --config outputs/configs/moabb_a100.materialized.yaml \
  --run-root "$PERSISTENT_ROOT/runs"
python -m neurotwin.cli report --run-dir "$PERSISTENT_ROOT/runs/moabb_a100_smoke"
bash scripts/package_a100_evidence_bundle.sh "$PERSISTENT_ROOT" outputs
'
