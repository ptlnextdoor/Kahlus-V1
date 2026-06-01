#!/usr/bin/env bash
set -euo pipefail
export COPYFILE_DISABLE=1

usage() {
  echo "usage: scripts/package_a100_evidence_bundle.sh /shared/persistent/neurotwin [output-dir]" >&2
}

if (($# < 1 || $# > 2)); then
  usage
  exit 2
fi

PERSISTENT_ROOT=$1
OUT_DIR=${2:-outputs}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -d "$PERSISTENT_ROOT" ]]; then
  echo "Persistent root does not exist: $PERSISTENT_ROOT" >&2
  exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to package the evidence bundle." >&2
  exit 2
fi

if [[ -f "$REPO_ROOT/COMMIT_HASH.txt" ]]; then
  FULL_SHA="$(head -n 1 "$REPO_ROOT/COMMIT_HASH.txt" | tr -d '[:space:]')"
else
  FULL_SHA="$(cd "$REPO_ROOT" && git rev-parse HEAD)"
fi
SHORT_SHA="${FULL_SHA:0:7}"
EVIDENCE_NAME="neurotwin-a100-results-$SHORT_SHA-evidence"
EVIDENCE_ZIP="$OUT_DIR/$EVIDENCE_NAME.zip"

mkdir -p "$OUT_DIR"

python3 "$SCRIPT_DIR/package_a100_evidence_bundle.py" \
  "$PERSISTENT_ROOT" \
  "$EVIDENCE_ZIP" \
  "$EVIDENCE_NAME" \
  "$REPO_ROOT" \
  "$FULL_SHA"
