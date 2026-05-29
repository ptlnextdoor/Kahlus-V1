#!/usr/bin/env bash
set -euo pipefail
export COPYFILE_DISABLE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SHORT_SHA="$(git rev-parse --short HEAD)"
FULL_SHA="$(git rev-parse HEAD)"
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

cat > "$HANDOFF_ROOT/README_HANDOFF.md" <<EOF
# NeuroTwin A100 Handoff

This handoff contains a runnable NeuroTwin A100 runner tarball for the Chapman preliminary infrastructure validation. It does not require private GitHub access.

This is minimal practical code visibility, not cryptographic source secrecy. The runner excludes git history, tests, research notes, paper drafts, raw data, prepared arrays, checkpoints, caches, local outputs, and secrets, but the Python runtime source needed to execute the job is present inside the runner tarball.

## Files

- \`$RUNNER_NAME.tar.gz\`: runner bundle to transfer to Chapman.
- \`COMMIT_HASH.txt\`: private repo commit used to build the runner.
- \`SHA256SUMS\`: checksums for this handoff folder.

## Verify This Handoff

\`\`\`bash
cat COMMIT_HASH.txt
shasum -a 256 -c SHA256SUMS
\`\`\`

## Transfer Through Raspberry Pi

Copy the runner tarball to the Raspberry Pi:

\`\`\`bash
scp $RUNNER_NAME.tar.gz <pi_user>@<raspberry_pi_host>:/tmp/
\`\`\`

From the Pi, copy it to the Chapman login node:

\`\`\`bash
scp /tmp/$RUNNER_NAME.tar.gz <chapman_user>@<chapman_login_host>:~/
\`\`\`

SSH from the Pi into the Chapman login node:

\`\`\`bash
ssh <chapman_user>@<chapman_login_host>
\`\`\`

The Pi is a transfer bridge only. Do not run Python training or submit Slurm jobs from the Pi outside an SSH session on the Chapman login node.

## Run On Chapman

\`\`\`bash
mkdir -p ~/neurotwin-a100
tar -xzf ~/$RUNNER_NAME.tar.gz -C ~/neurotwin-a100
cd ~/neurotwin-a100/$RUNNER_NAME
cat COMMIT_HASH.txt
sha256sum -c SHA256SUMS

conda env create -f environment-a100.yml
conda activate neurotwin-a100
python -m pip install -e '.[moabb,cluster]'

bash scripts/run_smoke.sh outputs/smoke
mkdir -p /path/to/shared/persistent/neurotwin
bash scripts/run_full.sh /path/to/shared/persistent/neurotwin
\`\`\`

The smoke test does not require an A100 or internet. MOABB preparation may need internet unless the MOABB cache is already populated. The training job reads prepared manifests from the persistent root and should not download data.

This first full run is an A100 infrastructure validation only. Do not interpret model quality from it.
EOF

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
