#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SHORT_SHA="$(git rev-parse --short HEAD)"
FULL_SHA="$(git rev-parse HEAD)"
OUT_DIR="outputs"
BUNDLE_NAME="neurotwin-a100-run-bundle-$SHORT_SHA"
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

mkdir -p "$OUT_DIR"
BUNDLE_ROOT="$STAGING/$BUNDLE_NAME"
mkdir -p "$BUNDLE_ROOT"

for path in README_RUN.md pyproject.toml environment-a100.yml requirements configs scripts src docs README.md AGENTS.md; do
  if [[ -e "$path" ]]; then
    cp -R "$path" "$BUNDLE_ROOT/"
  fi
done
printf '%s\n' "$FULL_SHA" > "$BUNDLE_ROOT/COMMIT_HASH.txt"

tar \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  --exclude='.mypy_cache' \
  --exclude='.ruff_cache' \
  -czf "$OUT_DIR/$BUNDLE_NAME.tar.gz" \
  -C "$STAGING" "$BUNDLE_NAME"
echo "bundle=$OUT_DIR/$BUNDLE_NAME.tar.gz"
echo "commit=$FULL_SHA"
