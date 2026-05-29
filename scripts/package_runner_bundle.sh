#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SHORT_SHA="$(git rev-parse --short HEAD)"
FULL_SHA="$(git rev-parse HEAD)"
OUT_DIR="outputs"
BUNDLE_NAME="neurotwin-a100-runner-$SHORT_SHA"
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

DIRTY=false
if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
  DIRTY=true
fi
if [[ "$DIRTY" == true && "${ALLOW_DIRTY_RUNNER_BUNDLE:-0}" != "1" ]]; then
  echo "Refusing to package a dirty worktree for commit $FULL_SHA." >&2
  echo "Commit or stash changes first, or set ALLOW_DIRTY_RUNNER_BUNDLE=1 to archive HEAD with dirty metadata." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
BUNDLE_ROOT="$STAGING/$BUNDLE_NAME"
ARCHIVE_PATHS=(
  README_RUN.md
  README.md
  pyproject.toml
  environment-a100.yml
  requirements/cluster-a100.txt
  configs/train/prepared_synthetic_debug.yaml
  configs/train/moabb_a100_smoke.yaml
  scripts/run_smoke.sh
  scripts/run_full.sh
  scripts/run_full.sbatch
  scripts/train_a100_inner.sh
  scripts/slurm/_train_a100_inner.sh
  scripts/prepare_moabb_benchmark.sh
  scripts/lib/moabb_prepare_common.sh
  scripts/package_runner_bundle.sh
  src
)

git archive --format=tar --prefix="$BUNDLE_NAME/" HEAD "${ARCHIVE_PATHS[@]}" | tar -xf - -C "$STAGING"

printf '%s\n' "$FULL_SHA" > "$BUNDLE_ROOT/COMMIT_HASH.txt"
{
  echo "commit=$FULL_SHA"
  echo "short_commit=$SHORT_SHA"
  echo "worktree_dirty=$DIRTY"
  echo "source=git archive HEAD"
  echo "bundle_type=runner"
} > "$BUNDLE_ROOT/BUNDLE_METADATA.txt"

(
  cd "$BUNDLE_ROOT"
  find . -type f \
    ! -name 'BUNDLE_MANIFEST.txt' \
    ! -name 'SHA256SUMS' \
    | sed 's#^\./##' \
    | LC_ALL=C sort > BUNDLE_MANIFEST.txt
  python3 - <<'PY' > SHA256SUMS
from hashlib import sha256
from pathlib import Path

paths = sorted(
    p for p in Path(".").rglob("*")
    if p.is_file() and p.name != "SHA256SUMS"
)
for path in paths:
    digest = sha256(path.read_bytes()).hexdigest()
    print(f"{digest}  {path.as_posix()}")
PY
)

tar \
  --exclude='.git' \
  --exclude='.git/*' \
  --exclude='.context' \
  --exclude='.context/*' \
  --exclude='tests' \
  --exclude='tests/*' \
  --exclude='docs/research' \
  --exclude='docs/research/*' \
  --exclude='docs/paper' \
  --exclude='docs/paper/*' \
  --exclude='graphify-out' \
  --exclude='graphify-out/*' \
  --exclude='outputs' \
  --exclude='outputs/*' \
  --exclude='runs' \
  --exclude='runs/*' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  --exclude='.mypy_cache' \
  --exclude='.ruff_cache' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.ckpt' \
  --exclude='*.npy' \
  --exclude='*.npz' \
  -czf "$OUT_DIR/$BUNDLE_NAME.tar.gz" \
  -C "$STAGING" "$BUNDLE_NAME"

python3 - "$OUT_DIR/$BUNDLE_NAME.tar.gz" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

path = Path(sys.argv[1])
print(f"bundle={path}")
print(f"bundle_sha256={sha256(path.read_bytes()).hexdigest()}")
PY
echo "commit=$FULL_SHA"
