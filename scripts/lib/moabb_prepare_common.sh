resolve_moabb_out_dir() {
  local prepared_name="$1"
  shift
  if (($#)); then
    printf '%s\n' "$1"
  elif [[ -n "${NEUROTWIN_DATA:-}" ]]; then
    printf '%s\n' "$NEUROTWIN_DATA/prepared/$prepared_name"
  elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "NEUROTWIN_DATA must be set or an output directory must be provided when running under SLURM." >&2
    return 2
  else
    printf '/tmp/neurotwin_%s\n' "$prepared_name"
  fi
}

print_moabb_prepare_summary() {
  python - "$1" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
audit = json.loads((root / "eval_audit.json").read_text(encoding="utf-8"))
print(f"summary_event_count={audit.get('event_count')}")
print(f"summary_window_count={audit.get('window_count')}")
print(f"summary_window_counts_by_split={audit.get('window_counts_by_split')}")
suite_path = root / "prepared_baseline_suite.json"
if suite_path.exists():
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    tasks = suite.get("tasks", {})
    for task_id in sorted(tasks):
        payload = tasks.get(task_id, {})
        if isinstance(payload, dict):
            print(f"summary_task_status_{task_id}={payload.get('status')}")
PY
}
