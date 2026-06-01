#!/usr/bin/env bash
set -euo pipefail
export COPYFILE_DISABLE=1

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
VERIFY_ROOT="$STAGING/verify"
ARCHIVE_PATHS=(
  README_HANDOFF.md.in
  README_RUN.md
  README_AGENT_DEPLOY.md
  README.md
  Dockerfile.a100
  pyproject.toml
  environment-a100.yml
  requirements/cluster-a100.txt
  configs/train/prepared_synthetic_debug.yaml
  configs/train/moabb_a100_smoke.yaml
  scripts/run_smoke.sh
  scripts/run_docker_6gpu.sh
  scripts/docker_a100_inner.sh
  scripts/docker_gpu_preflight.py
  scripts/run_full.sh
  scripts/run_full.sbatch
  scripts/train_a100_inner.sh
  scripts/package_a100_evidence_bundle.sh
  scripts/package_a100_evidence_bundle.py
  scripts/render_a100_handoff_readme.py
  scripts/slurm/_train_a100_inner.sh
  scripts/slurm/train_a100.sh
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

python3 "$SCRIPT_DIR/render_a100_handoff_readme.py" \
  --template "$BUNDLE_ROOT/README_HANDOFF.md.in" \
  --output "$BUNDLE_ROOT/README_HANDOFF.md" \
  --full-sha "$FULL_SHA" \
  --short-sha "$SHORT_SHA" \
  --runner-name "$BUNDLE_NAME" >/dev/null

python3 - "$BUNDLE_ROOT" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
forbidden_parts = {
    ".context",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "graphify-out",
    "outputs",
    "runs",
    "tests",
}
forbidden_prefixes = (
    ("docs", "paper"),
    ("docs", "research"),
)
forbidden_suffixes = {
    ".ckpt",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".pyc",
}
violations = []
for path in sorted(root.rglob("*")):
    rel = path.relative_to(root)
    parts = rel.parts
    if any(part.startswith("._") for part in parts):
        violations.append(rel.as_posix())
        continue
    if any(part in forbidden_parts for part in parts):
        violations.append(rel.as_posix())
        continue
    if any(parts[: len(prefix)] == prefix for prefix in forbidden_prefixes):
        violations.append(rel.as_posix())
        continue
    if path.is_file() and path.suffix in forbidden_suffixes:
        violations.append(rel.as_posix())

if violations:
    print("Refusing to package forbidden runner bundle paths:", file=sys.stderr)
    for violation in violations:
        print(f"  {violation}", file=sys.stderr)
    sys.exit(3)
PY

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

tar -czf "$OUT_DIR/$BUNDLE_NAME.tar.gz" -C "$STAGING" "$BUNDLE_NAME"

python3 - "$OUT_DIR/$BUNDLE_NAME.tar.gz" "$BUNDLE_NAME" "$BUNDLE_ROOT" <<'PY'
from pathlib import Path
import sys
import tarfile

archive = Path(sys.argv[1])
bundle_name = sys.argv[2]
bundle_root = Path(sys.argv[3])

expected_files = sorted(
    f"{bundle_name}/{path.relative_to(bundle_root).as_posix()}"
    for path in bundle_root.rglob("*")
    if path.is_file()
)
with tarfile.open(archive, "r:gz") as tar:
    actual_files = sorted(member.name for member in tar.getmembers() if member.isfile())

if actual_files != expected_files:
    missing = sorted(set(expected_files) - set(actual_files))
    extra = sorted(set(actual_files) - set(expected_files))
    if missing:
        print("archive is missing staged paths:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
    if extra:
        print("archive contains unstaged paths:", file=sys.stderr)
        for path in extra:
            print(f"  {path}", file=sys.stderr)
    sys.exit(4)
PY

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

actual_paths = sorted(
    path.as_posix()
    for path in Path(".").rglob("*")
    if path.is_file() and path.name != "SHA256SUMS"
)
expected_paths = sorted(expected)
if actual_paths != expected_paths:
    missing = sorted(set(expected_paths) - set(actual_paths))
    extra = sorted(set(actual_paths) - set(expected_paths))
    if missing:
        print("checksum manifest lists missing paths:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
    if extra:
        print("checksum manifest omits archive paths:", file=sys.stderr)
        for path in extra:
            print(f"  {path}", file=sys.stderr)
    sys.exit(4)

for rel in expected_paths:
    digest = sha256(Path(rel).read_bytes()).hexdigest()
    if digest != expected[rel]:
        print(f"checksum mismatch: {rel}", file=sys.stderr)
        sys.exit(5)
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
echo "commit=$FULL_SHA"
