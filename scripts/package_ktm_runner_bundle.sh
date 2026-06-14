#!/usr/bin/env bash
# Build the Kahlus v3 KTM A100 runner tarball (SYNTHETIC ONLY).
# Copies an allowlist from the WORKING TREE (not `git archive`), so it works before committing the
# Sprint 2B files. COMMIT_HASH.txt records the current HEAD; BUNDLE_METADATA records worktree_dirty.
set -euo pipefail
export COPYFILE_DISABLE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SHORT_SHA="$(git rev-parse --short HEAD)"
FULL_SHA="$(git rev-parse HEAD)"
OUT_DIR="${OUT_DIR:-outputs}"
BUNDLE_NAME="kahlus-ktm-a100-runner-$SHORT_SHA"
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

DIRTY=false
if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
  DIRTY=true
fi

mkdir -p "$OUT_DIR"
BUNDLE_ROOT="$STAGING/$BUNDLE_NAME"
VERIFY_ROOT="$STAGING/verify"
mkdir -p "$BUNDLE_ROOT"

ARCHIVE_PATHS=(
  pyproject.toml
  environment-ktm-a100.yml
  README_KTM_RUN.md
  README_KTM_HANDOFF.md.in
  configs/train/ktm_a100_micro.yaml
  configs/train/ktm_synthetic_smoke.yaml
  scripts/_bootstrap.py
  scripts/run_ktm_train.py
  scripts/run_ktm_smoke.sh
  scripts/docker_ktm_inner.sh
  scripts/run_docker_ktm.sh
  scripts/docker_gpu_preflight.py
  scripts/render_a100_handoff_readme.py
  scripts/package_ktm_runner_bundle.sh
  scripts/package_ktm_a100_handoff_zip.sh
  scripts/package_ktm_evidence_bundle.py
  scripts/slurm/train_ktm_a100.sh
  scripts/slurm/_train_ktm_a100_inner.sh
  src/neurotwin
)

# Copy the allowlist from the working tree, excluding caches / compiled / artifact noise.
tar -C "$REPO_ROOT" \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pyo' \
  --exclude='.mypy_cache' --exclude='.pytest_cache' --exclude='.ruff_cache' \
  --exclude='._*' \
  -cf - "${ARCHIVE_PATHS[@]}" | tar -xf - -C "$BUNDLE_ROOT"

# README_RUN.md is the canonical run book name inside the runner.
mv "$BUNDLE_ROOT/README_KTM_RUN.md" "$BUNDLE_ROOT/README_RUN.md"

printf '%s\n' "$FULL_SHA" > "$BUNDLE_ROOT/COMMIT_HASH.txt"
{
  echo "commit=$FULL_SHA"
  echo "short_commit=$SHORT_SHA"
  echo "worktree_dirty=$DIRTY"
  echo "source=worktree copy"
  echo "bundle_type=ktm-runner"
} > "$BUNDLE_ROOT/BUNDLE_METADATA.txt"

python3 "$SCRIPT_DIR/render_a100_handoff_readme.py" \
  --template "$BUNDLE_ROOT/README_KTM_HANDOFF.md.in" \
  --output "$BUNDLE_ROOT/README_HANDOFF.md" \
  --full-sha "$FULL_SHA" \
  --short-sha "$SHORT_SHA" \
  --runner-name "$BUNDLE_NAME" \
  --persistent-root-example "/raid/scratch/\$USER/kahlus-ktm-$SHORT_SHA" >/dev/null

# Refuse forbidden / secret-looking paths.
python3 - "$BUNDLE_ROOT" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
forbidden_parts = {".context", ".git", "__pycache__", "graphify-out", "outputs", "runs", "tests"}
forbidden_names = {"pw.txt", ".env"}
forbidden_suffixes = {".ckpt", ".npy", ".npz", ".pt", ".pth", ".pyc", ".pem", ".key"}
violations = []
for path in sorted(root.rglob("*")):
    rel = path.relative_to(root)
    parts = rel.parts
    if any(part.startswith("._") for part in parts):
        violations.append(rel.as_posix()); continue
    if forbidden_parts.intersection(parts):
        violations.append(rel.as_posix()); continue
    if forbidden_names.intersection(parts):
        violations.append(rel.as_posix()); continue
    if any(part.lower().startswith(".env.") for part in parts):
        violations.append(rel.as_posix()); continue
    if path.is_file() and path.suffix in forbidden_suffixes:
        violations.append(rel.as_posix())
if violations:
    print("Refusing to package forbidden KTM runner paths:", file=sys.stderr)
    for v in violations:
        print(f"  {v}", file=sys.stderr)
    sys.exit(3)
PY

(
  cd "$BUNDLE_ROOT"
  find . -type f ! -name 'BUNDLE_MANIFEST.txt' ! -name 'SHA256SUMS' \
    | sed 's#^\./##' | LC_ALL=C sort > BUNDLE_MANIFEST.txt
  python3 - <<'PY' > SHA256SUMS
from hashlib import sha256
from pathlib import Path
for path in sorted(p for p in Path(".").rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
    print(f"{sha256(path.read_bytes()).hexdigest()}  {path.as_posix()}")
PY
)

tar -czf "$OUT_DIR/$BUNDLE_NAME.tar.gz" -C "$STAGING" "$BUNDLE_NAME"

# Verify the extracted tarball matches its own SHA256SUMS.
mkdir -p "$VERIFY_ROOT"
tar -xzf "$OUT_DIR/$BUNDLE_NAME.tar.gz" -C "$VERIFY_ROOT"
(
  cd "$VERIFY_ROOT/$BUNDLE_NAME"
  python3 - <<'PY'
from hashlib import sha256
from pathlib import Path
import sys
expected = {}
for line in Path("SHA256SUMS").read_text(encoding="utf-8").splitlines():
    digest, rel = line.split("  ", 1)
    expected[rel] = digest
actual = sorted(p.as_posix() for p in Path(".").rglob("*") if p.is_file() and p.name != "SHA256SUMS")
if actual != sorted(expected):
    print("checksum manifest path mismatch", file=sys.stderr); sys.exit(4)
for rel, digest in expected.items():
    if sha256(Path(rel).read_bytes()).hexdigest() != digest:
        print(f"checksum mismatch: {rel}", file=sys.stderr); sys.exit(5)
PY
)

python3 - "$OUT_DIR/$BUNDLE_NAME.tar.gz" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys
path = Path(sys.argv[1])
print(f"bundle={path}")
print(f"bundle_sha256={sha256(path.read_bytes()).hexdigest()}")
PY
echo "commit=$FULL_SHA worktree_dirty=$DIRTY"
