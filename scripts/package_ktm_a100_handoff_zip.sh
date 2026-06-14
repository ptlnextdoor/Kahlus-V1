#!/usr/bin/env bash
# Build the Kahlus v3 KTM A100 handoff zip (SYNTHETIC ONLY):
#   kahlus-ktm-a100-handoff-<sha>/{COMMIT_HASH.txt, README_HANDOFF.md, SHA256SUMS, <runner>.tar.gz}
set -euo pipefail
export COPYFILE_DISABLE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SHORT_SHA="$(git rev-parse --short HEAD)"
FULL_SHA="$(git rev-parse HEAD)"
OUT_DIR="${OUT_DIR:-outputs}"
RUNNER_NAME="kahlus-ktm-a100-runner-$SHORT_SHA"
RUNNER_TARBALL="$OUT_DIR/$RUNNER_NAME.tar.gz"
HANDOFF_NAME="kahlus-ktm-a100-handoff-$SHORT_SHA"
HANDOFF_ZIP="$OUT_DIR/$HANDOFF_NAME.zip"
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

OUT_DIR="$OUT_DIR" bash "$SCRIPT_DIR/package_ktm_runner_bundle.sh"

if [[ ! -f "$RUNNER_TARBALL" ]]; then
  echo "Expected runner tarball not found: $RUNNER_TARBALL" >&2
  exit 3
fi

HANDOFF_ROOT="$STAGING/$HANDOFF_NAME"
mkdir -p "$HANDOFF_ROOT"
cp "$RUNNER_TARBALL" "$HANDOFF_ROOT/"
printf '%s\n' "$FULL_SHA" > "$HANDOFF_ROOT/COMMIT_HASH.txt"

python3 "$SCRIPT_DIR/render_a100_handoff_readme.py" \
  --template README_KTM_HANDOFF.md.in \
  --output "$HANDOFF_ROOT/README_HANDOFF.md" \
  --full-sha "$FULL_SHA" \
  --short-sha "$SHORT_SHA" \
  --runner-name "$RUNNER_NAME" \
  --persistent-root-example "/raid/scratch/\$USER/kahlus-ktm-$SHORT_SHA"

# The handoff README must byte-match the runner's rendered README_HANDOFF.md.
python3 - "$HANDOFF_ROOT/README_HANDOFF.md" "$RUNNER_TARBALL" "$RUNNER_NAME" <<'PY'
from pathlib import Path
import sys, tarfile
handoff_readme = Path(sys.argv[1]).read_text(encoding="utf-8")
with tarfile.open(sys.argv[2], "r:gz") as tar:
    member = tar.extractfile(f"{sys.argv[3]}/README_HANDOFF.md")
    if member is None:
        print("runner tarball missing README_HANDOFF.md", file=sys.stderr); sys.exit(4)
    if member.read().decode("utf-8") != handoff_readme:
        print("handoff README does not match runner README_HANDOFF.md", file=sys.stderr); sys.exit(4)
PY

( cd "$HANDOFF_ROOT"
  python3 - <<'PY' > SHA256SUMS
from hashlib import sha256
from pathlib import Path
for path in sorted(p for p in Path(".").iterdir() if p.is_file() and p.name != "SHA256SUMS"):
    print(f"{sha256(path.read_bytes()).hexdigest()}  {path.as_posix()}")
PY
)

python3 - "$HANDOFF_ROOT" "$HANDOFF_ZIP" "$HANDOFF_NAME" "$RUNNER_NAME" <<'PY'
from pathlib import Path
import sys, zipfile
handoff_root, zip_path, handoff_name, runner_name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
zip_path.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(handoff_root.iterdir()):
        if path.is_file():
            archive.write(path, f"{handoff_name}/{path.name}")
expected = {
    f"{handoff_name}/COMMIT_HASH.txt",
    f"{handoff_name}/README_HANDOFF.md",
    f"{handoff_name}/SHA256SUMS",
    f"{handoff_name}/{runner_name}.tar.gz",
}
with zipfile.ZipFile(zip_path, "r") as archive:
    actual = set(archive.namelist())
if actual != expected:
    print(f"handoff zip mismatch expected={sorted(expected)} actual={sorted(actual)}", file=sys.stderr)
    sys.exit(4)
PY

python3 - "$HANDOFF_ZIP" "$RUNNER_TARBALL" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys
print(f"runner_bundle={sys.argv[2]}")
print(f"runner_bundle_sha256={sha256(Path(sys.argv[2]).read_bytes()).hexdigest()}")
print(f"handoff_zip={sys.argv[1]}")
print(f"handoff_zip_sha256={sha256(Path(sys.argv[1]).read_bytes()).hexdigest()}")
PY
echo "commit=$FULL_SHA"
