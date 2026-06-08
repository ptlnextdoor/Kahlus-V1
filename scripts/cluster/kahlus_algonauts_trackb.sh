#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  PERSISTENT_ROOT=/raid/scratch/$USER/kahlus-algonauts-trackb-v1 bash scripts/cluster/kahlus_algonauts_trackb.sh prepare
  HOST_GPU_ID=5 bash scripts/cluster/kahlus_algonauts_trackb.sh debug
  HOST_GPU_IDS=0,1,2,3,4,5 bash scripts/cluster/kahlus_algonauts_trackb.sh sweep-seed <0|1|2>
  HOST_GPU_IDS=0,1,2,3,4,5 bash scripts/cluster/kahlus_algonauts_trackb.sh long
  bash scripts/cluster/kahlus_algonauts_trackb.sh status

This launcher starts detached tmux/Docker jobs and returns after launch.
Codex should monitor only until the safe-launch artifacts appear.
EOF
}

MODE="${1:-}"
SEED="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PERSISTENT_ROOT="${PERSISTENT_ROOT:-/raid/scratch/${USER}/kahlus-algonauts-trackb-v1}"
PREPARED_ROOT="${PREPARED_ROOT:-$PERSISTENT_ROOT/prepared/algonauts2025}"
CONFIG_ROOT="${CONFIG_ROOT:-$PERSISTENT_ROOT/inputs/configs}"
RUN_ROOT="${RUN_ROOT:-$PERSISTENT_ROOT/runs}"
LOG_ROOT="${LOG_ROOT:-$PERSISTENT_ROOT/logs}"
DOCKER_IMAGE="${DOCKER_IMAGE:-pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel}"
ALGONAUTS_RAW_ROOT="${ALGONAUTS_RAW_ROOT:-}"
WINDOW_LENGTH="${WINDOW_LENGTH:-128}"
STRIDE="${STRIDE:-64}"
DEBUG_STEPS="${DEBUG_STEPS:-500}"
DEBUG_BASELINE_STEPS="${DEBUG_BASELINE_STEPS:-3}"
SWEEP_STEPS="${SWEEP_STEPS:-10000}"
LONG_STEPS="${LONG_STEPS:-50000}"
SESSION_PREFIX="${SESSION_PREFIX:-kahlus-algonauts}"

if [[ -z "$MODE" || "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  usage
  exit 0
fi

mkdir -p "$PERSISTENT_ROOT" "$PREPARED_ROOT" "$CONFIG_ROOT" "$RUN_ROOT" "$LOG_ROOT"

require_abs() {
  local name=$1
  local value=$2
  if [[ -z "$value" || "$value" != /* ]]; then
    echo "$name must be an absolute path, got: $value" >&2
    exit 2
  fi
}

discover_algonauts_root() {
  if [[ -n "$ALGONAUTS_RAW_ROOT" ]]; then
    echo "$ALGONAUTS_RAW_ROOT"
    return 0
  fi
  local candidate
  for candidate in \
    "$PERSISTENT_ROOT/raw/algonauts2025_parent" \
    "/raid/scratch/$USER/neurotwin-algonauts2025" \
    "$PERSISTENT_ROOT/raw/algonauts2025" \
    "$PERSISTENT_ROOT/data/algonauts2025" \
    "/raid/scratch/$USER/neurotwin-algonauts2025/data" \
    "/raid/scratch/$USER/neurotwin-algonauts2025-v1/data" \
    "/raid/scratch/$USER/kahlus-algonauts-trackb-v1/raw/algonauts2025"; do
    if [[ -d "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

docker_mount_args() {
  local raw_root=$1
  printf '%q ' \
    -v "$REPO_ROOT:/workspace/repo" \
    -v "$PERSISTENT_ROOT:$PERSISTENT_ROOT"
  if [[ "$raw_root" != "$PERSISTENT_ROOT"* ]]; then
    printf '%q ' -v "$raw_root:$raw_root:ro"
  fi
}

docker_run_one_gpu() {
  local host_gpu=$1
  local raw_root=$2
  local command=$3
  # shellcheck disable=SC2046
  docker run --rm -i \
    --gpus "\"device=$host_gpu\"" \
    --ipc=host \
    --shm-size=64g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    $(docker_mount_args "$raw_root") \
    -w /workspace/repo \
    -e PYTHONPATH=src \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PERSISTENT_ROOT="$PERSISTENT_ROOT" \
    -e ALGONAUTS_RAW_ROOT="$raw_root" \
    -e PREPARED_ROOT="$PREPARED_ROOT" \
    -e CONFIG_ROOT="$CONFIG_ROOT" \
    -e RUN_ROOT="$RUN_ROOT" \
    -e WINDOW_LENGTH="$WINDOW_LENGTH" \
    -e STRIDE="$STRIDE" \
    -e DEBUG_STEPS="$DEBUG_STEPS" \
    -e DEBUG_BASELINE_STEPS="$DEBUG_BASELINE_STEPS" \
    -e SWEEP_STEPS="$SWEEP_STEPS" \
    -e LONG_STEPS="$LONG_STEPS" \
    "$DOCKER_IMAGE" bash -lc "$command"
}

docker_run_multi_gpu() {
  local host_gpus=$1
  local raw_root=$2
  local nproc=$3
  local command=$4
  # shellcheck disable=SC2046
  docker run --rm -i \
    --gpus "\"device=$host_gpus\"" \
    --ipc=host \
    --shm-size=128g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    $(docker_mount_args "$raw_root") \
    -w /workspace/repo \
    -e PYTHONPATH=src \
    -e CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((nproc - 1))")" \
    -e PERSISTENT_ROOT="$PERSISTENT_ROOT" \
    -e ALGONAUTS_RAW_ROOT="$raw_root" \
    -e PREPARED_ROOT="$PREPARED_ROOT" \
    -e CONFIG_ROOT="$CONFIG_ROOT" \
    -e RUN_ROOT="$RUN_ROOT" \
    -e WINDOW_LENGTH="$WINDOW_LENGTH" \
    -e STRIDE="$STRIDE" \
    -e LONG_STEPS="$LONG_STEPS" \
    "$DOCKER_IMAGE" bash -lc "$command"
}

write_status() {
  local state=$1
  python3 - "$PERSISTENT_ROOT/trackb_status.json" "$state" "$MODE" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "state": sys.argv[2],
    "mode": sys.argv[3],
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

prepare_inner() {
  local raw_root
  raw_root="$(discover_algonauts_root)" || {
    echo "No Algonauts raw root found. Set ALGONAUTS_RAW_ROOT to a verified local dataset/features root." >&2
    exit 3
  }
  require_abs ALGONAUTS_RAW_ROOT "$raw_root"
  python -m pip install -e ".[cluster]" >/tmp/kahlus_trackb_pip_install.log
  python -m neurotwin.cli data prepare \
    --dataset algonauts2025 \
    --root "$raw_root" \
    --out-dir "$PREPARED_ROOT" \
    --split official \
    --window-length "$WINDOW_LENGTH" \
    --stride "$STRIDE"
}

debug_inner() {
  prepare_inner
  mkdir -p "$PERSISTENT_ROOT/eval/algonauts_debug_paper_mode" "$CONFIG_ROOT"
  python -m neurotwin.cli eval suite \
    --suite neural_translation_v1 \
    --paper-mode \
    --seeds 0 1 2 \
    --event-manifest "$PREPARED_ROOT/event_manifest.json" \
    --split-manifest "$PREPARED_ROOT/split_manifest.json" \
    --window-length "$WINDOW_LENGTH" \
    --stride "$STRIDE" \
    --train-steps "$DEBUG_BASELINE_STEPS" \
    --out-dir "$PERSISTENT_ROOT/eval/algonauts_debug_paper_mode"
  python -m neurotwin.cli cluster materialize-config \
    --template /workspace/repo/configs/train/algonauts_pair_operator_debug.yaml \
    --prepared-root "$PREPARED_ROOT" \
    --out "$CONFIG_ROOT/algonauts_pair_operator_debug.materialized.yaml"
  python - <<'PY'
import os
from pathlib import Path
import yaml

path = Path(os.environ["CONFIG_ROOT"]) / "algonauts_pair_operator_debug.materialized.yaml"
cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
cfg["steps"] = int(os.environ.get("DEBUG_STEPS", "500"))
cfg["run_id"] = cfg["experiment"] = "algonauts_pair_operator_debug_gate"
path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY
  A100_PAPER_MODE_EVAL_DIR="$PERSISTENT_ROOT/eval/algonauts_debug_paper_mode" \
  A100_RUN_PAPER_MODE_IN_FULL=0 \
  bash /workspace/repo/scripts/slurm/_train_a100_inner.sh \
    "$CONFIG_ROOT/algonauts_pair_operator_debug.materialized.yaml" \
    "$RUN_ROOT" \
    1
}

sweep_materialize_inner() {
  if [[ -z "$SEED" ]]; then
    echo "sweep-materialize-inner requires seed argument" >&2
    exit 2
  fi
  prepare_inner
  local ablation_config_dir="$CONFIG_ROOT/sweep_seed$SEED"
  mkdir -p "$ablation_config_dir"
  python /workspace/repo/scripts/materialize_pair_operator_ablation_configs.py \
    --template /workspace/repo/configs/train/algonauts_pair_operator_ablation_array.yaml \
    --prepared-root "$PREPARED_ROOT" \
    --out-dir "$ablation_config_dir" \
    --seed "$SEED" \
    --steps "$SWEEP_STEPS" \
    --include current_neurotwin pair_operator_no_pair_state pair_operator_low_rank_pair_state pair_operator_pair_state_uncertainty pair_operator_full
}

sweep_ridge_inner() {
  if [[ -z "$SEED" ]]; then
    echo "sweep-ridge-inner requires seed argument" >&2
    exit 2
  fi
  python -m pip install -e ".[cluster]" >/tmp/kahlus_trackb_pip_install.log
  local sweep_root="$PERSISTENT_ROOT/sweep/seed$SEED"
  mkdir -p "$sweep_root/ridge_anchor"
  python -m neurotwin.cli eval suite \
    --suite neural_translation_v1 \
    --seed "$SEED" \
    --event-manifest "$PREPARED_ROOT/event_manifest.json" \
    --split-manifest "$PREPARED_ROOT/split_manifest.json" \
    --window-length "$WINDOW_LENGTH" \
    --stride "$STRIDE" \
    --train-steps "$DEBUG_BASELINE_STEPS" \
    --out-dir "$sweep_root/ridge_anchor"
}

sweep_arm_inner() {
  local arm="${ARM_NAME:-}"
  if [[ -z "$SEED" || -z "$arm" ]]; then
    echo "sweep-arm-inner requires SEED and ARM_NAME" >&2
    exit 2
  fi
  python -m pip install -e ".[cluster]" >/tmp/kahlus_trackb_pip_install.log
  local config="$CONFIG_ROOT/sweep_seed$SEED/$arm.seed$SEED.materialized.yaml"
  if [[ ! -f "$config" ]]; then
    echo "materialized config missing for arm=$arm seed=$SEED: $config" >&2
    exit 3
  fi
  A100_PAPER_MODE_EVAL_DIR="$PERSISTENT_ROOT/eval/algonauts_debug_paper_mode" \
  A100_RUN_PAPER_MODE_IN_FULL=0 \
  bash /workspace/repo/scripts/slurm/_train_a100_inner.sh "$config" "$RUN_ROOT" 1
}

long_inner() {
  prepare_inner
  mkdir -p "$CONFIG_ROOT"
  python -m neurotwin.cli cluster materialize-config \
    --template /workspace/repo/configs/train/algonauts_pair_operator_full.yaml \
    --prepared-root "$PREPARED_ROOT" \
    --out "$CONFIG_ROOT/algonauts_pair_operator_full.materialized.yaml"
  python - <<'PY'
import os
from pathlib import Path
import yaml

path = Path(os.environ["CONFIG_ROOT"]) / "algonauts_pair_operator_full.materialized.yaml"
cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
cfg["steps"] = int(os.environ.get("LONG_STEPS", "50000"))
cfg["run_id"] = cfg["experiment"] = "algonauts_pair_operator_full_long"
path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY
  A100_PAPER_MODE_EVAL_DIR="$PERSISTENT_ROOT/eval/algonauts_debug_paper_mode" \
  A100_RUN_PAPER_MODE_IN_FULL=0 \
  bash /workspace/repo/scripts/slurm/_train_a100_inner.sh \
    "$CONFIG_ROOT/algonauts_pair_operator_full.materialized.yaml" \
    "$RUN_ROOT" \
    6
}

launch_detached() {
  local session=$1
  shift
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 4
  fi
  tmux new-session -d -s "$session" "$*"
  echo "session=$session"
}

case "$MODE" in
  prepare)
    write_status "preparing"
    prepare_inner 2>&1 | tee "$LOG_ROOT/prepare.log"
    write_status "prepared"
    ;;
  debug)
    raw_root="$(discover_algonauts_root)" || {
      echo "No Algonauts raw root found. Set ALGONAUTS_RAW_ROOT before launching debug." >&2
      exit 3
    }
    require_abs ALGONAUTS_RAW_ROOT "$raw_root"
    if [[ -z "${HOST_GPU_ID:-}" ]]; then
      echo "HOST_GPU_ID is required for debug" >&2
      exit 2
    fi
    write_status "debug_launched"
    launch_detached "$SESSION_PREFIX-debug" \
      "cd '$REPO_ROOT' && PERSISTENT_ROOT='$PERSISTENT_ROOT' ALGONAUTS_RAW_ROOT='$raw_root' HOST_GPU_ID='$HOST_GPU_ID' bash scripts/cluster/kahlus_algonauts_trackb.sh debug-docker 2>&1 | tee '$LOG_ROOT/debug.log'"
    ;;
  debug-docker)
    raw_root="$(discover_algonauts_root)"
    docker_run_one_gpu "$HOST_GPU_ID" "$raw_root" "bash scripts/cluster/kahlus_algonauts_trackb.sh debug-inner"
    ;;
  debug-inner)
    debug_inner
    ;;
  sweep-seed)
    if [[ -z "$SEED" ]]; then
      echo "sweep-seed requires seed argument" >&2
      exit 2
    fi
    raw_root="$(discover_algonauts_root)" || {
      echo "No Algonauts raw root found. Set ALGONAUTS_RAW_ROOT before launching sweep." >&2
      exit 3
    }
    if [[ -z "${HOST_GPU_IDS:-}" ]]; then
      echo "HOST_GPU_IDS is required for sweep-seed" >&2
      exit 2
    fi
    write_status "sweep_seed${SEED}_launched"
    launch_detached "$SESSION_PREFIX-sweep-seed$SEED" \
      "cd '$REPO_ROOT' && PERSISTENT_ROOT='$PERSISTENT_ROOT' ALGONAUTS_RAW_ROOT='$raw_root' HOST_GPU_IDS='$HOST_GPU_IDS' SEED='$SEED' bash scripts/cluster/kahlus_algonauts_trackb.sh sweep-seed-docker '$SEED' 2>&1 | tee '$LOG_ROOT/sweep_seed${SEED}.log'"
    ;;
  sweep-seed-docker)
    raw_root="$(discover_algonauts_root)"
    IFS=, read -r -a gpu_ids <<<"$HOST_GPU_IDS"
    if [[ "${#gpu_ids[@]}" -ne 6 ]]; then
      echo "HOST_GPU_IDS must contain exactly six comma-separated A100 ids for sweep, got: $HOST_GPU_IDS" >&2
      exit 2
    fi
    docker_run_one_gpu "${gpu_ids[0]}" "$raw_root" "SEED='$SEED' bash scripts/cluster/kahlus_algonauts_trackb.sh sweep-materialize-inner '$SEED'"
    arms=(
      current_neurotwin
      pair_operator_no_pair_state
      pair_operator_low_rank_pair_state
      pair_operator_pair_state_uncertainty
      pair_operator_full
    )
    docker_run_one_gpu "${gpu_ids[0]}" "$raw_root" "SEED='$SEED' bash scripts/cluster/kahlus_algonauts_trackb.sh sweep-ridge-inner '$SEED'" \
      2>&1 | tee "$LOG_ROOT/sweep_seed${SEED}_ridge_anchor.log" &
    pids=("$!")
    for idx in "${!arms[@]}"; do
      gpu_index=$((idx + 1))
      arm="${arms[$idx]}"
      docker_run_one_gpu "${gpu_ids[$gpu_index]}" "$raw_root" "SEED='$SEED' ARM_NAME='$arm' bash scripts/cluster/kahlus_algonauts_trackb.sh sweep-arm-inner '$SEED'" \
        2>&1 | tee "$LOG_ROOT/sweep_seed${SEED}_${arm}.log" &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        failed=1
      fi
    done
    exit "$failed"
    ;;
  sweep-materialize-inner)
    SEED="${SEED:-${2:-}}"
    sweep_materialize_inner
    ;;
  sweep-ridge-inner)
    SEED="${SEED:-${2:-}}"
    sweep_ridge_inner
    ;;
  sweep-arm-inner)
    SEED="${SEED:-${2:-}}"
    sweep_arm_inner
    ;;
  gate)
    python "$REPO_ROOT/scripts/cluster/kahlus_algonauts_trackb_gate.py" \
      --root "$PERSISTENT_ROOT" \
      --out "$PERSISTENT_ROOT/strict_gate.json"
    ;;
  long)
    raw_root="$(discover_algonauts_root)" || {
      echo "No Algonauts raw root found. Set ALGONAUTS_RAW_ROOT before launching long run." >&2
      exit 3
    }
    if [[ -z "${HOST_GPU_IDS:-}" ]]; then
      echo "HOST_GPU_IDS is required for long" >&2
      exit 2
    fi
    IFS=, read -r -a gpu_ids <<<"$HOST_GPU_IDS"
    if [[ "${#gpu_ids[@]}" -ne 6 ]]; then
      echo "HOST_GPU_IDS must contain exactly six comma-separated A100 ids for long, got: $HOST_GPU_IDS" >&2
      exit 2
    fi
    write_status "long_launched"
    launch_detached "$SESSION_PREFIX-long" \
      "cd '$REPO_ROOT' && PERSISTENT_ROOT='$PERSISTENT_ROOT' ALGONAUTS_RAW_ROOT='$raw_root' HOST_GPU_IDS='$HOST_GPU_IDS' bash scripts/cluster/kahlus_algonauts_trackb.sh long-docker 2>&1 | tee '$LOG_ROOT/long.log'"
    ;;
  long-docker)
    raw_root="$(discover_algonauts_root)"
    docker_run_multi_gpu "$HOST_GPU_IDS" "$raw_root" 6 "bash scripts/cluster/kahlus_algonauts_trackb.sh long-inner"
    ;;
  long-inner)
    long_inner
    ;;
  status)
    echo "persistent_root=$PERSISTENT_ROOT"
    [[ -f "$PERSISTENT_ROOT/trackb_status.json" ]] && cat "$PERSISTENT_ROOT/trackb_status.json"
    tmux ls 2>/dev/null | grep "$SESSION_PREFIX" || true
    find "$PERSISTENT_ROOT" -maxdepth 4 -type f \( -name 'summary.json' -o -name 'metrics.csv' -o -name 'eval_audit.json' -o -name 'paper_mode_gate.json' -o -name 'strict_gate.json' \) | sort
    ;;
  *)
    usage
    exit 2
    ;;
esac
