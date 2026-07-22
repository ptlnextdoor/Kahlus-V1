#!/usr/bin/env bash
set -euo pipefail
export COPYFILE_DISABLE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SHORT_SHA="$(git rev-parse --short HEAD)"
FULL_SHA="$(git rev-parse HEAD)"
IMAGE_REF="${NEUROTWIN_DOCKER_IMAGE_REF:-ghcr.io/ptlnextdoor/kahlus-v1-a100-runtime:$SHORT_SHA}"
OUT_DIR="outputs"
HANDOFF_NAME="neurotwin-a100-nfc-docker-handoff-$SHORT_SHA"
HANDOFF_ZIP="$OUT_DIR/$HANDOFF_NAME.zip"
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

DIRTY=false
if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
  DIRTY=true
fi
if [[ "$DIRTY" == true && "${ALLOW_DIRTY_NFC_DOCKER_HANDOFF:-0}" != "1" ]]; then
  echo "Refusing to package an NFC Docker handoff from a dirty worktree for commit $FULL_SHA." >&2
  echo "Commit or stash changes first, or set ALLOW_DIRTY_NFC_DOCKER_HANDOFF=1 for local draft packaging." >&2
  exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to create the NFC Docker handoff zip." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
HANDOFF_ROOT="$STAGING/$HANDOFF_NAME"
mkdir -p "$HANDOFF_ROOT"
printf '%s\n' "$FULL_SHA" > "$HANDOFF_ROOT/COMMIT_HASH.txt"

python3 - "$FULL_SHA" "$SHORT_SHA" "$IMAGE_REF" deploy/a100/README_KRISH_AGENT_NFC_A100.md.in "$HANDOFF_ROOT/README_KRISH_AGENT.md" <<'PY'
from pathlib import Path
from string import Template
import sys

full_sha, short_sha, image_ref, template_path, output_path = sys.argv[1:6]
rendered = Template(Path(template_path).read_text(encoding="utf-8")).safe_substitute(
    FULL_SHA=full_sha,
    SHORT_SHA=short_sha,
    IMAGE_REF=image_ref,
)
Path(output_path).write_text(rendered, encoding="utf-8")
PY

python3 - "$HANDOFF_ROOT" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
forbidden_suffixes = (".py", ".pyc", ".whl", ".tar", ".gz", ".zip", ".pt", ".pth", ".npz", ".npy", ".pem", ".key")
forbidden_names = {".env", "pw.txt"}
violations = []
for path in sorted(root.rglob("*")):
    rel = path.relative_to(root)
    if path.is_dir() and rel.parts and rel.parts[0] == "src":
        violations.append(rel.as_posix())
    if path.is_file():
        lower_parts = {part.lower() for part in rel.parts}
        if forbidden_names & lower_parts:
            violations.append(rel.as_posix())
        if any(part.lower().startswith(".env.") for part in rel.parts):
            violations.append(rel.as_posix())
        if path.suffix in forbidden_suffixes or path.name.endswith(".tar.gz"):
            violations.append(rel.as_posix())
if violations:
    print("Refusing to package source-bearing or secret-like NFC Docker handoff paths:", file=sys.stderr)
    for violation in violations:
        print(f"  {violation}", file=sys.stderr)
    sys.exit(3)
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
    f"{handoff_name}/README_KRISH_AGENT.md",
    f"{handoff_name}/SHA256SUMS",
}
with zipfile.ZipFile(zip_path, "r") as archive:
    actual = set(archive.namelist())
if actual != expected:
    print("NFC Docker handoff zip content mismatch:", file=sys.stderr)
    print(f"expected={sorted(expected)}", file=sys.stderr)
    print(f"actual={sorted(actual)}", file=sys.stderr)
    sys.exit(4)
PY

python3 - "$HANDOFF_ZIP" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

zip_path = Path(sys.argv[1])
print(f"nfc_docker_handoff_zip={zip_path}")
print(f"nfc_docker_handoff_zip_sha256={sha256(zip_path.read_bytes()).hexdigest()}")
PY
echo "commit=$FULL_SHA"
echo "image_ref=$IMAGE_REF"
