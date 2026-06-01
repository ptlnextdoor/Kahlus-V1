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

Commit: \`$FULL_SHA\`

This handoff contains a runnable NeuroTwin A100 runner tarball for the Chapman preliminary infrastructure validation. It does not require private GitHub access.

This is minimal practical code visibility, not cryptographic source secrecy. The runner excludes git history, tests, research notes, paper drafts, raw data, prepared arrays, checkpoints, caches, local outputs, and secrets, but the Python runtime source needed to execute the job is present inside the runner tarball.

## Purpose

This run is meant to prove that the codeless A100 handoff can be unpacked, verified, installed, smoke-tested, prepared on MOABB, audited for leakage/window counts, trained through the Docker 6-GPU cluster path, and reported from a persistent Chapman root.

## Not A Claim

This is not a scientific result, not a model-superiority claim, not a 3-seed paper-mode report, and not a clinical claim. The expected \`summary.json\` keeps \`scientific_claim_allowed=false\`.

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

Extract and verify the runner:

\`\`\`bash
mkdir -p ~/neurotwin-a100
tar -xzf ~/$RUNNER_NAME.tar.gz -C ~/neurotwin-a100
cd ~/neurotwin-a100/$RUNNER_NAME
cat COMMIT_HASH.txt
sha256sum -c SHA256SUMS
\`\`\`

Primary Docker 6-GPU path:

\`\`\`bash
export PERSISTENT_ROOT=/raid/scratch/\$USER/neurotwin-$SHORT_SHA
TARGET_GPUS=6 bash scripts/run_docker_6gpu.sh "\$PERSISTENT_ROOT" all
\`\`\`

The helper launches Docker with the runner mounted at \`/workspace/repo\`, persistent outputs under \`/raid/scratch/\$USER/neurotwin-$SHORT_SHA\`, and refuses to train unless six CUDA devices are visible. An automated deployment agent should follow \`README_AGENT_DEPLOY.md\`; \`Dockerfile.a100\` is included for agents that prefer to build a local image before running.

\`\`\`bash
docker run --rm -it --gpus all --ipc=host \\
  -v "\$PWD":/workspace/repo \\
  -v "\$PERSISTENT_ROOT":"\$PERSISTENT_ROOT" \\
  -w /workspace/repo \\
  -e PERSISTENT_ROOT="\$PERSISTENT_ROOT" \\
  -e NEUROTWIN_DATA="\$PERSISTENT_ROOT" \\
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel bash
\`\`\`

The smoke test does not require an A100 or internet. MOABB preparation may need internet unless the MOABB cache is already populated. The training job reads prepared manifests from the persistent root and should not download data.

## Conda And Slurm Alternative

Use this only when \`conda\` and \`sbatch\` are available:

\`\`\`bash
conda env create -f environment-a100.yml
conda activate neurotwin-a100
python -m pip install -e '.[moabb,cluster]'

bash scripts/run_smoke.sh outputs/smoke
mkdir -p /path/to/shared/persistent/neurotwin
bash scripts/run_full.sh /path/to/shared/persistent/neurotwin
\`\`\`

For a 6-GPU Slurm follow-up after the materialized config exists:

\`\`\`bash
export NEUROTWIN_DATA=/path/to/shared/persistent/neurotwin
export RUN_ROOT="\$NEUROTWIN_DATA/runs"
RUN_ROOT="\$RUN_ROOT" \\
sbatch --ntasks-per-node=6 --gres=gpu:a100:6 \\
  scripts/slurm/train_a100.sh outputs/configs/moabb_a100.materialized.yaml
\`\`\`

## Docker Command Details

The packaged helper is the recommended Docker fallback when \`conda\` or \`sbatch\` are missing:

\`\`\`bash
export PERSISTENT_ROOT=/raid/scratch/\$USER/neurotwin-$SHORT_SHA
TARGET_GPUS=6 bash scripts/run_docker_6gpu.sh "\$PERSISTENT_ROOT" all
\`\`\`

For a one-GPU diagnostic, pass one visible GPU id and override the process count:

\`\`\`bash
ALLOW_FEWER_GPUS=1 TARGET_GPUS=1 bash scripts/run_docker_6gpu.sh "\$PERSISTENT_ROOT" <gpu_id>
\`\`\`

In that diagnostic mode the helper passes Docker \`--gpus "device=<gpu_id>"\`. Do not treat a one-GPU diagnostic as the requested 6-GPU run.

The helper runs this host command:

\`\`\`bash
docker run --rm -it --gpus all --ipc=host \\
  -v "\$PWD":/workspace/repo \\
  -v "\$PERSISTENT_ROOT":"\$PERSISTENT_ROOT" \\
  -w /workspace/repo \\
  -e PERSISTENT_ROOT="\$PERSISTENT_ROOT" \\
  -e NEUROTWIN_DATA="\$PERSISTENT_ROOT" \\
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel bash
\`\`\`

Inside the container, the helper executes:

\`\`\`bash
python -m pip install -e '.[moabb,cluster]'
bash scripts/run_smoke.sh outputs/smoke
bash scripts/prepare_moabb_benchmark.sh "\$PERSISTENT_ROOT/prepared/moabb_benchmark"
python -m neurotwin.cli eval audit \\
  --suite neural_translation_v1 \\
  --event-manifest "\$PERSISTENT_ROOT/prepared/moabb_benchmark/event_manifest.json" \\
  --split-manifest "\$PERSISTENT_ROOT/prepared/moabb_benchmark/split_manifest.json" \\
  --window-length 128 \\
  --stride 128 \\
  --out-dir "\$PERSISTENT_ROOT/prepared/moabb_benchmark" \\
  --require-windows
python -m neurotwin.cli cluster materialize-config \\
  --template configs/train/moabb_a100_smoke.yaml \\
  --prepared-root "\$PERSISTENT_ROOT/prepared/moabb_benchmark" \\
  --out outputs/configs/moabb_a100.materialized.yaml
python -m neurotwin.cli cluster preflight \\
  --config outputs/configs/moabb_a100.materialized.yaml \\
  --run-root "\$PERSISTENT_ROOT/runs" \\
  --require-cuda \\
  --require-prepared-windows \\
  --expect-window-count 18144 \\
  --expect-split-windows train:12096,val:2016,test:4032
torchrun --standalone --nproc_per_node=6 \\
  -m neurotwin.cli train \\
  --config outputs/configs/moabb_a100.materialized.yaml \\
  --run-root "\$PERSISTENT_ROOT/runs"
python -m neurotwin.cli report --run-dir "\$PERSISTENT_ROOT/runs/moabb_a100_smoke"
bash scripts/package_a100_evidence_bundle.sh "\$PERSISTENT_ROOT" outputs
\`\`\`

## Expected Outputs

Prepared artifacts:

\`\`\`text
\$PERSISTENT_ROOT/prepared/moabb_benchmark/eval_audit.json
\$PERSISTENT_ROOT/prepared/moabb_benchmark/data_manifest.json
\$PERSISTENT_ROOT/prepared/moabb_benchmark/event_manifest.json
\$PERSISTENT_ROOT/prepared/moabb_benchmark/split_manifest.json
\$PERSISTENT_ROOT/prepared/moabb_benchmark/leakage_report.json
\`\`\`

Run artifacts:

\`\`\`text
\$PERSISTENT_ROOT/gpu_preflight.json
\$PERSISTENT_ROOT/runs/moabb_a100_smoke/config.yaml
\$PERSISTENT_ROOT/runs/moabb_a100_smoke/environment.json
\$PERSISTENT_ROOT/runs/moabb_a100_smoke/checkpoint.pt
\$PERSISTENT_ROOT/runs/moabb_a100_smoke/checkpoint_best.pt
\$PERSISTENT_ROOT/runs/moabb_a100_smoke/metrics.json
\$PERSISTENT_ROOT/runs/moabb_a100_smoke/metrics.csv
\$PERSISTENT_ROOT/runs/moabb_a100_smoke/metrics.jsonl
\$PERSISTENT_ROOT/runs/moabb_a100_smoke/summary.json
\$PERSISTENT_ROOT/runs/moabb_a100_smoke/tables/
\$PERSISTENT_ROOT/runs/moabb_a100_smoke/figures/
\`\`\`

Expected audit gate:

\`\`\`text
eval_audit_passed=True
window_count=18144
window_counts_by_split=train:12096,val:2016,test:4032
\`\`\`

## Send Back Evidence

After the run, package only reviewable outputs:

\`\`\`bash
bash scripts/package_a100_evidence_bundle.sh "\$PERSISTENT_ROOT" outputs
\`\`\`

This excludes checkpoints, raw prepared arrays, runner tarballs, zip artifacts, passwords, API keys, SSH keys, \`.env*\` files, and private keys.

## Known Limitations

- Docker fallback does not submit Slurm; it runs directly inside Docker with the GPU list passed to \`scripts/run_docker_6gpu.sh\`.
- MOABB preparation may need internet unless the cache is already populated.
- The short full run is expected to report \`completed_steps=50\`, \`real_data_smoke=true\`, and \`scientific_claim_allowed=false\`.
- MOABB task labels are intentionally removed from prepared event metadata before persistence; forbidden model-visible event metadata fields are \`label\`, \`target\`, \`target_label\`, \`task_label\`, and \`diagnosis\`.

Do not interpret model quality from this first full run.
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
