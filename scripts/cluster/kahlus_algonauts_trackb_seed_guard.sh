#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-0}"
PERSISTENT_ROOT="${PERSISTENT_ROOT:-/raid/scratch/${USER}/kahlus-algonauts-trackb-v1}"
PREPARED_ROOT="${PREPARED_ROOT:-$PERSISTENT_ROOT/prepared/algonauts2025_f71c1ba}"
REPO_ROOT="${REPO_ROOT:-$HOME/Kahlus-V1-trackb}"
CONFIG_ROOT="${CONFIG_ROOT:-$PERSISTENT_ROOT/inputs/configs}"
RUN_ROOT="${RUN_ROOT:-$PERSISTENT_ROOT/runs}"
LOG_ROOT="${LOG_ROOT:-$PERSISTENT_ROOT/logs}"
SESSION_PREFIX="${SESSION_PREFIX:-kahlus-algonauts}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MAX_POLLS="${MAX_POLLS:-0}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-1024}"
GPU_UTIL_LIMIT="${GPU_UTIL_LIMIT:-10}"

STATUS_PATH="$PERSISTENT_ROOT/sweep_seed${SEED}_guard_status.json"
LOG_PATH="$LOG_ROOT/sweep_seed${SEED}_guard.log"
SWEEP_SESSION="$SESSION_PREFIX-sweep-seed$SEED"
GUARD_SESSION="$SESSION_PREFIX-sweep-seed${SEED}-guard"

mkdir -p "$PERSISTENT_ROOT" "$LOG_ROOT" "$RUN_ROOT" "$CONFIG_ROOT"

write_status() {
  local state="$1"
  local message="${2:-}"
  local host_gpu_ids="${3:-}"
  python3 - "$STATUS_PATH" "$state" "$message" "$host_gpu_ids" "$SEED" "$PERSISTENT_ROOT" "$PREPARED_ROOT" "$SWEEP_SESSION" <<'PY'
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
    "seed": int(sys.argv[5]),
    "persistent_root": sys.argv[6],
    "prepared_root": sys.argv[7],
    "sweep_session": sys.argv[8],
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
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

if [[ "$SEED" != "0" && "$SEED" != "1" && "$SEED" != "2" ]]; then
  echo "seed must be 0, 1, or 2; got $SEED" >&2
  exit 2
fi

if ! debug_gate_passed; then
  write_status "blocked_debug_gate" "debug_gate.json is missing or did not pass"
  exit 3
fi

poll_count=0
write_status "waiting_for_gpus" "waiting for exactly six idle A100 GPUs"

while true; do
  if tmux has-session -t "$SWEEP_SESSION" 2>/dev/null; then
    write_status "existing_sweep_session" "sweep session already exists; refusing duplicate launch"
    exit 0
  fi

  ids="$(idle_gpu_ids)"
  IFS=, read -r -a gpu_ids <<<"$ids"
  if [[ -n "$ids" && "${#gpu_ids[@]}" -eq 6 ]]; then
    write_status "launching" "six idle A100 GPUs found; launching sweep seed $SEED" "$ids"
    (
      cd "$REPO_ROOT"
      PERSISTENT_ROOT="$PERSISTENT_ROOT" \
      PREPARED_ROOT="$PREPARED_ROOT" \
      CONFIG_ROOT="$CONFIG_ROOT" \
      RUN_ROOT="$RUN_ROOT" \
      LOG_ROOT="$LOG_ROOT" \
      HOST_GPU_IDS="$ids" \
      SESSION_PREFIX="$SESSION_PREFIX" \
      bash scripts/cluster/kahlus_algonauts_trackb.sh sweep-seed "$SEED"
    )
    write_status "launched" "sweep seed $SEED launch command returned" "$ids"
    exit 0
  fi

  write_status "waiting_for_gpus" "idle GPUs available: ${ids:-none}; need exactly six" "$ids"
  poll_count=$((poll_count + 1))
  if [[ "$MAX_POLLS" != "0" && "$poll_count" -ge "$MAX_POLLS" ]]; then
    write_status "max_polls_reached" "guard reached MAX_POLLS=$MAX_POLLS before launch" "$ids"
    exit 4
  fi
  sleep "$POLL_SECONDS"
done 2>&1 | tee -a "$LOG_PATH"
