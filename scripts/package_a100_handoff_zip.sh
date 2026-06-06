#!/usr/bin/env bash
set -euo pipefail
export COPYFILE_DISABLE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SHORT_SHA="$(git rev-parse --short HEAD)"
FULL_SHA="$(git rev-parse HEAD)"
IMAGE_REF="${NEUROTWIN_DOCKER_IMAGE_REF:-pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel}"
OUT_DIR="outputs"
RUNNER_NAME="neurotwin-a100-runner-$SHORT_SHA"
RUNNER_TARBALL="$OUT_DIR/$RUNNER_NAME.tar.gz"
HANDOFF_NAME="neurotwin-a100-handoff-$SHORT_SHA"
HANDOFF_ZIP="$OUT_DIR/$HANDOFF_NAME.zip"
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to create the handoff zip." >&2
  exit 2
fi

bash scripts/package_runner_bundle.sh

if [[ ! -f "$RUNNER_TARBALL" ]]; then
  echo "Expected runner tarball not found: $RUNNER_TARBALL" >&2
  exit 3
fi

HANDOFF_ROOT="$STAGING/$HANDOFF_NAME"
mkdir -p "$HANDOFF_ROOT"
cp "$RUNNER_TARBALL" "$HANDOFF_ROOT/"
printf '%s\n' "$FULL_SHA" > "$HANDOFF_ROOT/COMMIT_HASH.txt"

python3 scripts/render_a100_handoff_readme.py \
  --template README_HANDOFF.md.in \
  --output "$HANDOFF_ROOT/README_HANDOFF.md" \
  --full-sha "$FULL_SHA" \
  --short-sha "$SHORT_SHA" \
  --runner-name "$RUNNER_NAME"

python3 - "$FULL_SHA" "$SHORT_SHA" "$RUNNER_NAME" "$RUNNER_NAME.tar.gz" "$IMAGE_REF" \
  scripts/a100_krish_agent_autorun.sh.in "$HANDOFF_ROOT/run_krish_agent.sh" <<'PY'
from pathlib import Path
import sys

full_sha, short_sha, runner_name, runner_tarball, image_ref, template_path, output_path = sys.argv[1:8]
rendered = (
    Path(template_path)
    .read_text(encoding="utf-8")
    .replace("__FULL_SHA__", full_sha)
    .replace("__SHORT_SHA__", short_sha)
    .replace("__RUNNER_NAME__", runner_name)
    .replace("__RUNNER_TARBALL__", runner_tarball)
    .replace("__IMAGE_REF__", image_ref)
)
Path(output_path).write_text(rendered, encoding="utf-8")
Path(output_path).chmod(0o755)
PY

python3 - "$HANDOFF_ROOT/README_HANDOFF.md" "$RUNNER_TARBALL" "$RUNNER_NAME" <<'PY'
from pathlib import Path
import sys
import tarfile

handoff_readme = Path(sys.argv[1]).read_text(encoding="utf-8")
runner_tarball = Path(sys.argv[2])
runner_name = sys.argv[3]
with tarfile.open(runner_tarball, "r:gz") as tar:
    member = tar.extractfile(f"{runner_name}/README_HANDOFF.md")
    if member is None:
        print("runner tarball is missing README_HANDOFF.md", file=sys.stderr)
        sys.exit(4)
    runner_readme = member.read().decode("utf-8")
if runner_readme != handoff_readme:
    print("handoff README does not match runner README_HANDOFF.md", file=sys.stderr)
    sys.exit(4)
PY

(
  cd "$HANDOFF_ROOT"
  python3 - <<'PY' > SHA256SUMS
from hashlib import sha256
from pathlib import Path

for path in sorted(p for p in Path(".").iterdir() if p.is_file() and p.name != "SHA256SUMS"):
    print(f"{sha256(path.read_bytes()).hexdigest()}  {path.as_posix()}")
PY
)

python3 - "$HANDOFF_ROOT" "$HANDOFF_ZIP" "$HANDOFF_NAME" <<'PY'
from pathlib import Path
import sys
import zipfile

handoff_root = Path(sys.argv[1])
zip_path = Path(sys.argv[2])
handoff_name = sys.argv[3]

zip_path.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(handoff_root.iterdir()):
        if path.is_file():
            archive.write(path, f"{handoff_name}/{path.name}")

expected = {
    f"{handoff_name}/COMMIT_HASH.txt",
    f"{handoff_name}/README_HANDOFF.md",
    f"{handoff_name}/run_krish_agent.sh",
    f"{handoff_name}/SHA256SUMS",
    f"{handoff_name}/{handoff_name.replace('handoff', 'runner')}.tar.gz",
}
with zipfile.ZipFile(zip_path, "r") as archive:
    actual = set(archive.namelist())
if actual != expected:
    print("handoff zip content mismatch:", file=sys.stderr)
    print(f"expected={sorted(expected)}", file=sys.stderr)
    print(f"actual={sorted(actual)}", file=sys.stderr)
    sys.exit(4)
PY

python3 - "$HANDOFF_ZIP" "$RUNNER_TARBALL" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

zip_path = Path(sys.argv[1])
runner_path = Path(sys.argv[2])
print(f"runner_bundle={runner_path}")
print(f"runner_bundle_sha256={sha256(runner_path.read_bytes()).hexdigest()}")
print(f"handoff_zip={zip_path}")
print(f"handoff_zip_sha256={sha256(zip_path.read_bytes()).hexdigest()}")
PY
echo "commit=$FULL_SHA"
echo "image_ref=$IMAGE_REF"
