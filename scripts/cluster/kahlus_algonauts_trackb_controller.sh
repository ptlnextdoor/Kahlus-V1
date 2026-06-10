#!/usr/bin/env bash
set -euo pipefail

PERSISTENT_ROOT="${PERSISTENT_ROOT:-/raid/scratch/${USER}/kahlus-algonauts-trackb-v1}"
PREPARED_ROOT="${PREPARED_ROOT:-$PERSISTENT_ROOT/prepared/algonauts2025_f71c1ba}"
REPO_ROOT="${REPO_ROOT:-$HOME/Kahlus-V1-trackb}"
CONFIG_ROOT="${CONFIG_ROOT:-$PERSISTENT_ROOT/inputs/configs}"
RUN_ROOT="${RUN_ROOT:-$PERSISTENT_ROOT/runs}"
LOG_ROOT="${LOG_ROOT:-$PERSISTENT_ROOT/logs}"
SESSION_PREFIX="${SESSION_PREFIX:-kahlus-algonauts}"
POLL_SECONDS="${POLL_SECONDS:-600}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-1024}"
GPU_UTIL_LIMIT="${GPU_UTIL_LIMIT:-10}"

STATUS_PATH="$PERSISTENT_ROOT/trackb_controller_status.json"
LOG_PATH="$LOG_ROOT/trackb_controller.log"

mkdir -p "$PERSISTENT_ROOT" "$LOG_ROOT" "$RUN_ROOT" "$CONFIG_ROOT"

write_status() {
  local state="$1"
  local message="${2:-}"
  local host_gpu_ids="${3:-}"
  python3 - "$STATUS_PATH" "$state" "$message" "$host_gpu_ids" "$PERSISTENT_ROOT" "$PREPARED_ROOT" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "state": sys.argv[2],
    "message": sys.argv[3],
    "host_gpu_ids": sys.argv[4],
    "persistent_root": sys.argv[5],
    "prepared_root": sys.argv[6],
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

tmux_session_exists_exact() {
  local session="$1"
  tmux list-sessions -F '#S' 2>/dev/null | grep -Fx -- "$session" >/dev/null
}

debug_gate_passed() {
  python3 - "$PERSISTENT_ROOT/debug_gate.json" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
raise SystemExit(0 if payload.get("passed") is True else 1)
PY
}

seed_complete() {
  local seed="$1"
  python3 - "$PERSISTENT_ROOT" "$seed" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
seed = int(sys.argv[2])
arms = (
    "current_neurotwin",
    "pair_operator_no_pair_state",
    "pair_operator_low_rank_pair_state",
    "pair_operator_pair_state_uncertainty",
    "pair_operator_full",
)

def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

ridge = root / "sweep" / f"seed{seed}" / "ridge_anchor" / "prepared_baseline_suite.json"
if not ridge.exists():
    raise SystemExit(1)
try:
    ridge_payload = json.loads(ridge.read_text(encoding="utf-8"))
except json.JSONDecodeError:
    raise SystemExit(1)
ridge_task = (ridge_payload.get("tasks") or {}).get("stimulus_to_fmri_response") or {}
ridge_metrics = (ridge_task.get("metrics_by_model") or {}).get("linear_ridge") or {}
if not finite(ridge_metrics.get("pearsonr")):
    raise SystemExit(1)

for arm in arms:
    summary = root / "runs" / f"{arm}_seed{seed}" / "summary.json"
    if not summary.exists():
        raise SystemExit(1)
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise SystemExit(1)
    if payload.get("status") != "completed_prepared_training":
        raise SystemExit(1)
    if payload.get("quarantined_tasks"):
        raise SystemExit(1)
    task = {}
    for row in payload.get("task_results") or []:
        if isinstance(row, dict) and row.get("task_id") == "stimulus_to_fmri_response":
            task = row
            break
    if not task and payload.get("task_id") == "stimulus_to_fmri_response":
        task = payload
    if not task or not finite(task.get("test_pearsonr")):
        raise SystemExit(1)
raise SystemExit(0)
PY
}

seed_attempt_started() {
  local seed="$1"
  find "$LOG_ROOT" -maxdepth 1 -type f \( \
    -name "sweep_seed${seed}.log" -o \
    -name "sweep_seed${seed}_ridge_anchor.log" -o \
    -name "sweep_seed${seed}_current_neurotwin.log" -o \
    -name "sweep_seed${seed}_pair_operator_no_pair_state.log" -o \
    -name "sweep_seed${seed}_pair_operator_low_rank_pair_state.log" -o \
    -name "sweep_seed${seed}_pair_operator_pair_state_uncertainty.log" -o \
    -name "sweep_seed${seed}_pair_operator_full.log" \
  \) -size +0 -print -quit | grep -q .
}

strict_gate_passed() {
  python3 - "$PERSISTENT_ROOT/strict_gate.json" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
raise SystemExit(0 if payload.get("passed") is True else 1)
PY
}

idle_gpu_ids() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | python3 -c '
from __future__ import annotations

import sys

memory_limit = int(sys.argv[1])
util_limit = int(sys.argv[2])
rows = []
for line in sys.stdin:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        continue
    try:
        index = int(parts[0])
        memory = int(parts[1])
        util = int(parts[2])
    except ValueError:
        continue
    if memory <= memory_limit and util <= util_limit:
        rows.append((memory, util, index))
print(",".join(str(index) for memory, util, index in sorted(rows)[:6]))
' "$GPU_MEMORY_LIMIT_MIB" "$GPU_UTIL_LIMIT"
}

start_seed_guard() {
  local seed="$1"
  local guard_session="$SESSION_PREFIX-guard-sweep-seed$seed"
  if tmux_session_exists_exact "$guard_session"; then
    return 0
  fi
  tmux new-session -d -s "$guard_session" \
    "cd '$REPO_ROOT' && PERSISTENT_ROOT='$PERSISTENT_ROOT' PREPARED_ROOT='$PREPARED_ROOT' POLL_SECONDS=300 SESSION_PREFIX='$SESSION_PREFIX' bash scripts/cluster/kahlus_algonauts_trackb_seed_guard.sh '$seed'"
}

launch_long_if_gpus_idle() {
  local ids="$1"
  (
    cd "$REPO_ROOT"
    PERSISTENT_ROOT="$PERSISTENT_ROOT" \
    PREPARED_ROOT="$PREPARED_ROOT" \
    CONFIG_ROOT="$CONFIG_ROOT" \
    RUN_ROOT="$RUN_ROOT" \
    LOG_ROOT="$LOG_ROOT" \
    HOST_GPU_IDS="$ids" \
    SESSION_PREFIX="$SESSION_PREFIX" \
    bash scripts/cluster/kahlus_algonauts_trackb.sh long
  )
}

write_status "started" "Track B controller started"

while true; do
  if ! debug_gate_passed; then
    write_status "blocked_debug_gate" "debug_gate.json is missing or did not pass"
    exit 3
  fi

  for seed in 0 1 2; do
    if seed_complete "$seed"; then
      continue
    fi
    sweep_session="$SESSION_PREFIX-sweep-seed$seed"
    guard_session="$SESSION_PREFIX-guard-sweep-seed$seed"
    if tmux_session_exists_exact "$sweep_session"; then
      write_status "sweep_running" "sweep seed $seed is running in $sweep_session"
      sleep "$POLL_SECONDS"
      continue 2
    fi
    if seed_attempt_started "$seed"; then
      write_status "sweep_incomplete_after_attempt" "sweep seed $seed has logs but no complete artifacts; stopping for inspection"
      exit 5
    fi
    if tmux_session_exists_exact "$guard_session"; then
      write_status "guard_waiting" "sweep seed $seed guard is active in $guard_session"
      sleep "$POLL_SECONDS"
      continue 2
    fi
    start_seed_guard "$seed"
    write_status "guard_started" "started sweep seed $seed guard in $guard_session"
    sleep "$POLL_SECONDS"
    continue 2
  done

  write_status "all_sweeps_complete" "all three sweep seeds completed; running strict gate"
  if ! (
    cd "$REPO_ROOT"
    PERSISTENT_ROOT="$PERSISTENT_ROOT" \
    PREPARED_ROOT="$PREPARED_ROOT" \
    CONFIG_ROOT="$CONFIG_ROOT" \
    RUN_ROOT="$RUN_ROOT" \
    LOG_ROOT="$LOG_ROOT" \
    bash scripts/cluster/kahlus_algonauts_trackb.sh gate
  ); then
    write_status "strict_gate_failed" "strict_gate.json did not pass; long run is blocked"
    exit 4
  fi
  if ! strict_gate_passed; then
    write_status "strict_gate_failed" "strict_gate.json did not pass; long run is blocked"
    exit 4
  fi

  if tmux_session_exists_exact "$SESSION_PREFIX-long"; then
    write_status "long_running" "long run session already exists: $SESSION_PREFIX-long"
    exit 0
  fi

  ids="$(idle_gpu_ids)"
  IFS=, read -r -a gpu_ids <<<"$ids"
  if [[ -n "$ids" && "${#gpu_ids[@]}" -eq 6 ]]; then
    write_status "long_launching" "strict gate passed; launching long run" "$ids"
    launch_long_if_gpus_idle "$ids"
    write_status "long_launched" "long run launch command returned" "$ids"
    exit 0
  fi

  write_status "waiting_for_long_gpus" "strict gate passed; idle GPUs available: ${ids:-none}; need exactly six" "$ids"
  sleep "$POLL_SECONDS"
done 2>&1 | tee -a "$LOG_PATH"
