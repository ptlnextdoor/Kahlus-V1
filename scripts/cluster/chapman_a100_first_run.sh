#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: scripts/cluster/chapman_a100_first_run.sh /shared/persistent/neurotwin" >&2
}

if (($# != 1)); then
  usage
  exit 2
fi

PERSISTENT_ROOT=$1
if [[ "$PERSISTENT_ROOT" != /* ]]; then
  echo "Persistent root must be an absolute shared filesystem path, got: $PERSISTENT_ROOT" >&2
  exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is required; run this from a Chapman login shell with SLURM available." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export NEUROTWIN_DATA="$PERSISTENT_ROOT"
export MOABB_DATA="$NEUROTWIN_DATA/moabb"
export BIDS_ROOT="$NEUROTWIN_DATA/bids"
export RUN_ROOT="$NEUROTWIN_DATA/runs"
EXPECTED_WINDOW_COUNT="${EXPECTED_WINDOW_COUNT:-18144}"

mkdir -p logs "$NEUROTWIN_DATA" "$MOABB_DATA" "$BIDS_ROOT" "$RUN_ROOT"

echo "step=prepare_moabb_benchmark"
bash scripts/prepare_moabb_benchmark.sh

echo "step=verify_window_gate"
python3 - "$NEUROTWIN_DATA/prepared/moabb_benchmark/eval_audit.json" "$EXPECTED_WINDOW_COUNT" <<'PY'
import json
import sys
from pathlib import Path

audit_path = Path(sys.argv[1])
expected = int(sys.argv[2])
payload = json.loads(audit_path.read_text(encoding="utf-8"))
window_count = int(payload.get("window_count", 0))
counts = payload.get("window_counts_by_split", {})
missing = [name for name in ("train", "val", "test") if int(counts.get(name, 0)) <= 0]
if not payload.get("passed"):
    raise SystemExit(f"eval audit failed: {payload.get('violations')}")
if window_count != expected:
    raise SystemExit(f"expected window_count={expected}, got {window_count}")
if missing:
    raise SystemExit("missing nonzero split windows: " + ",".join(missing))
print("eval_audit_passed=True")
print(f"window_count={window_count}")
print(
    "window_counts_by_split="
    + ",".join(f"{name}:{int(counts.get(name, 0))}" for name in ("train", "val", "test"))
)
PY

echo "step=materialize_chapman_config"
cp configs/train/moabb_a100_smoke.yaml configs/train/moabb_a100_chapman.yaml
python3 - <<'PY'
import os
from pathlib import Path

cfg = Path("configs/train/moabb_a100_chapman.yaml")
root = Path(os.environ["NEUROTWIN_DATA"]).resolve()
text = cfg.read_text(encoding="utf-8")
text = text.replace(
    "event_manifest: /path/to/moabb_prepared/event_manifest.json",
    f"event_manifest: {root}/prepared/moabb_benchmark/event_manifest.json",
)
text = text.replace(
    "split_manifest: /path/to/moabb_prepared/split_manifest.json",
    f"split_manifest: {root}/prepared/moabb_benchmark/split_manifest.json",
)
cfg.write_text(text, encoding="utf-8")
PY

echo "step=doctor"
PYTHONPATH=src python3 -m neurotwin.cli doctor

echo "step=preflight_without_cuda"
PYTHONPATH=src python3 -m neurotwin.cli cluster preflight \
  --config configs/train/moabb_a100_chapman.yaml \
  --run-root "$RUN_ROOT" \
  --require-prepared-windows

echo "step=dry_run"
PYTHONPATH=src python3 -m neurotwin.cli train --dry-run --config configs/train/moabb_a100_chapman.yaml

echo "step=submit_one_a100_job"
RUN_ROOT="$RUN_ROOT" sbatch scripts/slurm/train_a100.sh configs/train/moabb_a100_chapman.yaml
