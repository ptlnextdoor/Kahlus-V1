# NeuroTwin A100 Run Bundle

This bundle runs one controlled NeuroTwin MOABB infrastructure validation on an A100 cluster. It verifies imports, data preparation, leakage/window gates, CUDA visibility, prepared-manifest training, checkpoint writing, and report generation. It is not a scientific result and not the 3-seed acceptance run.

## What The Run Does

The smoke command validates local package, CLI, synthetic data, prepared training, and reporting. The full command validates the real MOABB preparation path and launches one A100 infrastructure job after exact window-count and cluster preflight gates pass.

## Bundle And Commit

Primary handoff is a runner tarball. The friend running Chapman does not need GitHub access.

Build the bundle from a clean committed checkout on the packaging machine:

```bash
bash scripts/package_runner_bundle.sh
ls outputs/neurotwin-a100-runner-*.tar.gz
```

The archive includes `COMMIT_HASH.txt`, `BUNDLE_MANIFEST.txt`, and `SHA256SUMS` so it maps back to the private repo commit and can be checked after transfer. It excludes `.git/`, git history, tests, research notes, paper drafts, graph output, raw data, prepared arrays, checkpoints, caches, local outputs, `.context/`, and local machine paths.

This is minimal practical code visibility, not cryptographic source secrecy. The runner contains Python runtime source because Chapman has to execute it. A determined operator with filesystem access can still inspect shipped Python files.

An internal full-source bundle is also available:

```bash
bash scripts/package_run_bundle.sh
```

Use the runner bundle for the friend-facing preliminary Chapman run.

If repo access is available, the equivalent source checkout is:

```bash
git clone <PRIVATE_REPO_URL>
cd Kahlus-V1
git checkout <COMMIT_HASH_FROM_HANDOFF>
git rev-parse HEAD
```

Use the exact commit hash supplied in the handoff message. Do not run from an edited worktree for the first A100 validation.

## Raspberry Pi Handoff Path

Use the Raspberry Pi only as a Chapman-network bridge and file shuttle. Do not build the bundle on the Pi, do not run Python training commands on the Pi, and do not submit Slurm from the Pi except through an SSH session on the Chapman login node.

From the packaging machine, copy the bundle to the Pi:

```bash
scp outputs/neurotwin-a100-runner-<short_sha>.tar.gz \
  <pi_user>@<raspberry_pi_host>:/tmp/
```

From the Pi, copy the bundle to the Chapman login node:

```bash
scp /tmp/neurotwin-a100-runner-<short_sha>.tar.gz \
  <chapman_user>@<chapman_login_host>:~/
```

SSH from the Pi into the Chapman login node and unpack there:

```bash
ssh <chapman_user>@<chapman_login_host>
mkdir -p ~/neurotwin-a100
tar -xzf ~/neurotwin-a100-runner-<short_sha>.tar.gz -C ~/neurotwin-a100
cd ~/neurotwin-a100/neurotwin-a100-runner-<short_sha>
cat COMMIT_HASH.txt
sha256sum -c SHA256SUMS
```

All remaining commands in this file run on the Chapman login node, not on the Pi.

## Environment

Recommended conda/mamba setup:

```bash
conda env create -f environment-a100.yml
conda activate neurotwin-a100
python -m pip install -e '.[moabb,cluster]'
```

Pip fallback when the cluster already provides CUDA-enabled PyTorch:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
python -m pip install -e '.[moabb,cluster]'
python -m pip install -r requirements/cluster-a100.txt
```

Expected baseline:

```text
Python: 3.10
CUDA: 12.1-compatible driver/runtime
PyTorch: >=2.2 with CUDA enabled
Major libraries: MOABB, MNE, MNE-BIDS, NumPy, pandas, PyYAML, SciPy, scikit-learn, tensorboard, wandb
```

## Tiny Smoke Test

No A100 or internet is required. This should finish in under 5-10 minutes:

```bash
bash scripts/run_smoke.sh outputs/smoke
```

Expected smoke outputs:

```text
outputs/smoke/prepared/event_manifest.json
outputs/smoke/prepared/split_manifest.json
outputs/smoke/runs/prepared_synthetic_debug/summary.json
outputs/smoke/runs/prepared_synthetic_debug/metrics.json
outputs/smoke/runs/prepared_synthetic_debug/tables/metrics_flat.csv
outputs/smoke/runs/prepared_synthetic_debug/figures/metric_summary.json
```

Smoke succeeds when the script prints `smoke_status=completed`.

## Optional RunPod A100 Rehearsal

Before Chapman, a short RunPod rehearsal can validate CUDA and the Slurm wrapper path. Keep it under the explicit budget cap:

```bash
export RUNPOD_MAX_BUDGET_USD=5
bash scripts/cluster/runpod_a100_rehearsal.sh /workspace/neurotwin_data
```

This must run inside a RunPod A100 pod from a clean checkout. A non-A100 GPU does not count as a passed A100 rehearsal.

## Full A100 Infrastructure Validation

Required resources:

```text
GPU: 1x A100 80GB
CPU: 16 cores
RAM: 128G
Wall time: 02:00:00 for the first infrastructure validation
```

Prepare a persistent shared root first:

```bash
mkdir -p /path/to/shared/persistent/neurotwin
bash scripts/run_full.sh /path/to/shared/persistent/neurotwin
```

Good root examples:

```bash
bash scripts/run_full.sh /shared/scratch/$USER/neurotwin
bash scripts/run_full.sh /data/$USER/neurotwin
```

Bad root examples:

```bash
bash scripts/run_full.sh /tmp/neurotwin
bash scripts/run_full.sh ./outputs/full
bash scripts/run_full.sh /Users/<local_user>/neurotwin
```

`scripts/run_full.sh` refuses `/tmp`, `/private/tmp`, `/var/tmp`, relative paths, repo-local paths, local laptop paths, missing roots, unwritable persistent logs, missing manifests after preparation, bad window counts, and failed preflight. It creates:

```text
/path/to/shared/persistent/neurotwin/moabb
/path/to/shared/persistent/neurotwin/bids
/path/to/shared/persistent/neurotwin/prepared
/path/to/shared/persistent/neurotwin/runs
/path/to/shared/persistent/neurotwin/logs
```

It materializes:

```text
outputs/configs/moabb_a100.materialized.yaml
```

with absolute manifest paths under:

```text
/path/to/shared/persistent/neurotwin/prepared/moabb_benchmark/
```

Generated cluster configs are intentionally written under `outputs/configs/`, not tracked `configs/train/`.
The exact MOABB gate is enforced by:

```bash
PYTHONPATH=src python3 -m neurotwin.cli cluster preflight \
  --config outputs/configs/moabb_a100.materialized.yaml \
  --run-root /path/to/shared/persistent/neurotwin/runs \
  --require-prepared-windows \
  --expect-window-count 18144 \
  --expect-split-windows train:12096,val:2016,test:4032
```

If Chapman requires a Slurm account, partition, or QoS, set them as environment variables before `run_full.sh`. Leave them unset if Chapman defaults are sufficient.

```bash
export SBATCH_PARTITION=gpu
export SBATCH_ACCOUNT=<your_chapman_account>
export SBATCH_QOS=<optional_qos>
bash scripts/run_full.sh /absolute/shared/persistent/neurotwin
```

The scripts pass these values to `sbatch` as command-line flags. They are not embedded in `#SBATCH` lines.

## Data And Internet

Dataset: MOABB `BNCI2014_001` with `LeftRightImagery`.

Data preparation may need internet if the MOABB cache is not already populated. Training itself must not download data; it reads prepared manifests from the persistent root.

No checkpoints, API keys, private datasets, or raw public neural data are required in the repo.

## Expected Full Outputs

Prepared data:

```text
$NEUROTWIN_DATA/prepared/moabb_benchmark/data_manifest.json
$NEUROTWIN_DATA/prepared/moabb_benchmark/split_manifest.json
$NEUROTWIN_DATA/prepared/moabb_benchmark/event_manifest.json
$NEUROTWIN_DATA/prepared/moabb_benchmark/leakage_report.json
$NEUROTWIN_DATA/prepared/moabb_benchmark/eval_audit.json
```

Required window gate:

```text
eval_audit_passed=True
window_count=18144
window_counts_by_split=train:12096,val:2016,test:4032
```

Run outputs:

```text
$NEUROTWIN_DATA/runs/moabb_a100_smoke/config.yaml
$NEUROTWIN_DATA/runs/moabb_a100_smoke/environment.json
$NEUROTWIN_DATA/runs/moabb_a100_smoke/checkpoint.pt
$NEUROTWIN_DATA/runs/moabb_a100_smoke/checkpoint_best.pt
$NEUROTWIN_DATA/runs/moabb_a100_smoke/metrics.json
$NEUROTWIN_DATA/runs/moabb_a100_smoke/metrics.csv
$NEUROTWIN_DATA/runs/moabb_a100_smoke/metrics.jsonl
$NEUROTWIN_DATA/runs/moabb_a100_smoke/summary.json
$NEUROTWIN_DATA/runs/moabb_a100_smoke/tables/metrics_flat.csv
$NEUROTWIN_DATA/runs/moabb_a100_smoke/figures/metric_summary.json
$NEUROTWIN_DATA/logs/neurotwin-a100-full-<jobid>.out
$NEUROTWIN_DATA/logs/neurotwin-a100-full-<jobid>.err
```

If the run directory already exists, NeuroTwin reuses the same run id. For a clean rerun, move or archive the previous `$NEUROTWIN_DATA/runs/moabb_a100_smoke` directory first.

## Success Condition

The first A100 validation succeeds only when:

- Slurm starts a job on an A100 allocation.
- `nt doctor` reports CUDA available and device count greater than zero inside the job.
- `nt cluster preflight --require-cuda --require-prepared-windows` passes.
- The prepared window gate is exactly `18144` total windows with `12096/2016/4032` train/val/test windows.
- Training writes checkpoints, metrics, and summary under `$NEUROTWIN_DATA/runs`.

Do not interpret model quality from this run. Scientific interpretation requires at least 3 real MOABB seeds, baseline reports, and `nt report --compare`.

## Resume And Safe Rerun

To resume manually after a failed training job:

```bash
export RUN_ROOT=/path/to/shared/persistent/neurotwin/runs
PYTHONPATH=src python3 -m neurotwin.cli train \
  --config outputs/configs/moabb_a100.materialized.yaml \
  --run-root "$RUN_ROOT" \
  --resume "$RUN_ROOT/moabb_a100_smoke/checkpoint.pt"
```

To safely rerun the full infrastructure validation, keep the prepared data and archive the previous run directory:

```bash
mv /path/to/shared/persistent/neurotwin/runs/moabb_a100_smoke \
   /path/to/shared/persistent/neurotwin/runs/moabb_a100_smoke.previous
bash scripts/run_full.sh /path/to/shared/persistent/neurotwin
```
