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

DIRTY=false
if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
  DIRTY=true
fi
if [[ "$DIRTY" == true && "${ALLOW_DIRTY_BUNDLE:-0}" != "1" ]]; then
  echo "Refusing to package a dirty worktree for commit $FULL_SHA." >&2
  echo "Commit or stash changes first, or set ALLOW_DIRTY_BUNDLE=1 to archive HEAD with dirty metadata." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
BUNDLE_ROOT="$STAGING/$BUNDLE_NAME"
ARCHIVE_PATHS=(
  README_RUN.md
  pyproject.toml
  environment-a100.yml
  requirements
  configs
  scripts
  src
  docs
  README.md
  AGENTS.md
)
TAR_EXCLUDES=(
  --exclude='.git'
  --exclude='.git/*'
  --exclude='.context'
  --exclude='.context/*'
  --exclude='outputs'
  --exclude='outputs/*'
  --exclude='runs'
  --exclude='runs/*'
  --exclude='data'
  --exclude='data/*'
  --exclude='*/__pycache__'
  --exclude='*.pyc'
  --exclude='.pytest_cache'
  --exclude='.mypy_cache'
  --exclude='.ruff_cache'
  --exclude='*.pt'
  --exclude='*.pth'
  --exclude='*.ckpt'
  --exclude='*.npy'
  --exclude='*.npz'
)

git archive --format=tar --prefix="$BUNDLE_NAME/" HEAD "${ARCHIVE_PATHS[@]}" | tar -xf - -C "$STAGING"
printf '%s\n' "$FULL_SHA" > "$BUNDLE_ROOT/COMMIT_HASH.txt"
{
  echo "commit=$FULL_SHA"
  echo "short_commit=$SHORT_SHA"
  echo "worktree_dirty=$DIRTY"
  echo "source=git archive HEAD"
} > "$BUNDLE_ROOT/BUNDLE_METADATA.txt"

tar \
  "${TAR_EXCLUDES[@]}" \
  -czf "$OUT_DIR/$BUNDLE_NAME.tar.gz" \
  -C "$STAGING" "$BUNDLE_NAME"
echo "bundle=$OUT_DIR/$BUNDLE_NAME.tar.gz"
echo "commit=$FULL_SHA"
