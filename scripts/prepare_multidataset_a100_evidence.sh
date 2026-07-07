#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-${PREPARED_OUT_DIR:-/tmp/kahlus_multidataset_a100_evidence}}"
ROOT_ARG=()
if [[ -n "${MULTIDATASET_ROOT:-}" ]]; then
  ROOT_ARG=(--root "${MULTIDATASET_ROOT}")
fi

# ponytail: dataset roots are persistent shared cache; never delete them here.
echo "dataset_persistence=keep_raw_dataset_roots_never_delete"
mkdir -p "${OUT_DIR}"

PYTHONPATH="${PYTHONPATH:-src}" python3 -m neurotwin.cli data prepare \
  --dataset multidataset_a100 \
  --split subject \
  "${ROOT_ARG[@]}" \
  --out-dir "${OUT_DIR}" \
  --max-trials "${MAX_RECORDS_PER_DATASET:-16}" \
  --window-length "${WINDOW_LENGTH:-128}" \
  --stride "${STRIDE:-128}"

mkdir -p "${OUT_DIR}/configs"
PYTHONPATH="${PYTHONPATH:-src}" python3 - <<'PY' "${OUT_DIR}"
from pathlib import Path
import sys

import yaml

out = Path(sys.argv[1]).resolve()
template_path = Path("configs/train/kahlus_multidataset_a100_evidence.yaml")
config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
config["data"]["event_manifest"] = str(out / "event_manifest.json")
config["data"]["split_manifest"] = str(out / "split_manifest.json")
config.setdefault("evidence", {})["prepared_root"] = str(out)
(out / "configs" / "kahlus_multidataset_a100_evidence.materialized.yaml").write_text(
    yaml.safe_dump(config, sort_keys=False),
    encoding="utf-8",
)
PY

PYTHONPATH="${PYTHONPATH:-src}" python3 -m neurotwin.cli eval audit \
  --event-manifest "${OUT_DIR}/event_manifest.json" \
  --split-manifest "${OUT_DIR}/split_manifest.json" \
  --window-length "${WINDOW_LENGTH:-128}" \
  --stride "${STRIDE:-128}" \
  --out-dir "${OUT_DIR}/eval_audit" \
  --require-windows

PYTHONPATH="${PYTHONPATH:-src}" python3 -m neurotwin.cli eval suite \
  --suite neural_translation_v1 \
  --event-manifest "${OUT_DIR}/event_manifest.json" \
  --split-manifest "${OUT_DIR}/split_manifest.json" \
  --window-length "${WINDOW_LENGTH:-128}" \
  --stride "${STRIDE:-128}" \
  --out-dir "${OUT_DIR}/baseline_suite" \
  --baseline-models persistence linear_ridge autoregressive_ridge tiny_ssm transformer tcn random_permutation time_shift_control patient_session_nuisance \
  --max-windows-per-split "${MAX_WINDOWS_PER_SPLIT:-128}" \
  --seed "${SEED:-0}"

echo "materialized_config=${OUT_DIR}/configs/kahlus_multidataset_a100_evidence.materialized.yaml"
