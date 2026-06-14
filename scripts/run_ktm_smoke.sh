#!/usr/bin/env bash
# Local CPU smoke for the Kahlus v3 KTM training harness (SYNTHETIC ONLY, no GPU, no A100).
# Mirrors scripts/run_smoke.sh: trains a tiny synthetic micro-run on CPU and asserts the output
# bundle is present and honestly gated. Prints `smoke_status=completed` on success.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${1:-outputs/smoke}"
CONFIG="${KTM_SMOKE_CONFIG:-configs/train/ktm_synthetic_smoke.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" scripts/run_ktm_train.py --out-dir "$OUT_DIR" --mode cpu_smoke --config "$CONFIG"

"$PYTHON_BIN" - "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
required = [
    "metrics.json", "baseline_table.json", "baseline_table.csv", "evidence_gate.json",
    "model_card.json", "data_card.json", "run_config.json", "failure_reasons.json",
    "environment.json", "progress.jsonl", "run_status.json",
]
missing = [name for name in required if not (out / name).is_file()]
if missing:
    print(f"smoke_status=failed missing={missing}", file=sys.stderr)
    raise SystemExit(1)

status = json.loads((out / "run_status.json").read_text())
if status.get("status") != "completed":
    print(f"smoke_status=failed run_status={status.get('status')}", file=sys.stderr)
    raise SystemExit(1)

gate = json.loads((out / "evidence_gate.json").read_text())
if gate["claim_scope"] != "synthetic_ktm_training_harness":
    print(f"smoke_status=failed unexpected_scope={gate['claim_scope']}", file=sys.stderr)
    raise SystemExit(1)

metrics = json.loads((out / "metrics.json").read_text())
if metrics["recovery_claim_allowed"]:
    print("smoke_status=failed recovery_claim_unexpectedly_allowed", file=sys.stderr)
    raise SystemExit(1)

env = json.loads((out / "environment.json").read_text())
assert "torch" in env and "version" in env["torch"], "environment.json missing torch fields"

print(f"smoke_status=completed out_dir={out}")
print(f"harness_scope={gate['claim_scope']} harness_allowed={gate['scientific_claim_allowed']}")
print(f"recovery_claim_allowed={metrics['recovery_claim_allowed']} (must be False)")
PY
