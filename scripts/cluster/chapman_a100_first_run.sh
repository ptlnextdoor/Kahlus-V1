#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: scripts/cluster/chapman_a100_first_run.sh /shared/persistent/neurotwin" >&2
}

if (($# != 1)); then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

exec "$REPO_ROOT/scripts/run_full.sh" "$@"
