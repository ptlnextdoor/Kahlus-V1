# NeuroTwin A100 Runner

This runner executes one controlled NeuroTwin MOABB infrastructure validation on an A100 cluster. It verifies imports, data preparation, leakage/window gates, CUDA visibility, prepared-manifest training, checkpoint writing, and report generation.

This is not a scientific result and not the 3-seed acceptance run.

## Purpose And Non-Purpose

Purpose: prove that this exact commit can run the codeless A100 path end to end on Chapman infrastructure, including checksum verification, environment install, MOABB preparation, eval audit, cluster preflight, one short prepared training run, and report generation.

Non-purpose: this package does not prove model superiority, paper readiness, clinical utility, or a 3-seed scientific result. It also does not hide source cryptographically; it is a practical runner bundle with the Python source required for execution.

## Operator Workflow

Run these commands on the Chapman login node after the runner tarball has been transferred there:

```bash
mkdir -p ~/neurotwin-a100
tar -xzf ~/neurotwin-a100-runner-<short_sha>.tar.gz -C ~/neurotwin-a100
cd ~/neurotwin-a100/neurotwin-a100-runner-<short_sha>
cat COMMIT_HASH.txt
sha256sum -c SHA256SUMS
```

### Primary Docker 6-GPU Path

Use this path when the machine has Docker with NVIDIA GPU support. It does not require `conda` or `sbatch`. The launcher defaults to host GPUs `0,1,2,3,4,5`, maps them to container devices `cuda:0` through `cuda:5`, probes CUDA inside the container, and refuses to train unless exactly six CUDA devices are visible.

```bash
export HOST_GPU_IDS=0,1,2,3,4,5
export GPU_COUNT=6
export NPROC_PER_NODE=6
export CONTAINER_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export A100_CONFIG_TEMPLATE=configs/train/moabb_a100.yaml
export A100_RUN_ID=moabb_a100
export PERSISTENT_ROOT=/raid/scratch/$USER/neurotwin-<short_sha>
bash scripts/run_docker_6gpu.sh "$PERSISTENT_ROOT"
```

The launcher auto-generates `DOCKER_LOG_PATH`, writes it to `$PERSISTENT_ROOT/docker_run.env`, and tees output to a current-run log named `neurotwin-a100-docker-<generated>.log`. Do not bypass `scripts/run_docker_6gpu.sh` for the full Docker run; it owns the log and evidence metadata contract.

An automated deployment agent should follow `README_AGENT_DEPLOY.md`. The runner also includes `Dockerfile.a100` as a dependency/runtime image helper. It does not hide source code; this runner still ships the runtime Python source required to execute.

Six-GPU Docker preflight:

```bash
HOST_GPU_IDS=0,1,2,3,4,5
docker run --rm --gpus "\"device=${HOST_GPU_IDS}\"" \
  --ipc=host --shm-size=64g \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash -lc 'nvidia-smi && python - <<'"'"'PY'"'"'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 6
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY'
```

The exact inside-container sequence lives in `scripts/docker_a100_inner.sh`; deployment-agent details live in `README_AGENT_DEPLOY.md`. The full Docker run command remains `bash scripts/run_docker_6gpu.sh "$PERSISTENT_ROOT"`, which writes `docker_run.env`, the Docker log, `gpu_preflight.json`, run outputs, and the evidence bundle.
If `A100_CONFIG_TEMPLATE` is unset, the helper keeps the short `configs/train/moabb_a100_smoke.yaml` infrastructure validation behavior. Set `A100_CONFIG_TEMPLATE=configs/train/moabb_a100.yaml` and `A100_RUN_ID=moabb_a100` for the long 6-GPU MOABB training lane.

The current training path supports single-node DDP through `torchrun`, `LOCAL_RANK`, `RANK`, `WORLD_SIZE`, `torch.cuda.set_device(local_rank)`, and PyTorch `DistributedDataParallel` wrapping. The code uses container-local CUDA device indexes and does not hard-code host GPU IDs.

### Conda And Slurm Alternative

Use this path when `conda` and `sbatch` are available and Chapman should manage the allocation.

```bash
conda env create -f environment-a100.yml
conda activate neurotwin-a100
python -m pip install -e '.[moabb,cluster]'
```

Run the local smoke test before submitting the A100 job:

```bash
bash scripts/run_smoke.sh outputs/smoke
```

Prepare a persistent shared root and launch the 1-GPU infrastructure validation:

```bash
mkdir -p /path/to/shared/persistent/neurotwin
bash scripts/run_full.sh /path/to/shared/persistent/neurotwin
```

All commands after unpacking run on the Chapman login node, not on a laptop or Raspberry Pi.

## Bundle Contents

The runner includes `COMMIT_HASH.txt`, `BUNDLE_MANIFEST.txt`, and `SHA256SUMS` so the transfer can be checked before execution. The runner contains the Python runtime source needed to execute on Chapman. It does not include git history, tests, paper drafts, research notes, graph output, raw data, prepared arrays, checkpoints, caches, local outputs, or `.context/`.

MOABB task labels are intentionally not persisted in prepared event metadata. The forbidden model-visible event metadata fields are `label`, `target`, `target_label`, `task_label`, and `diagnosis`; the prepared eval audit must pass without a post-hoc sanitize step.

## Raspberry Pi Handoff Path

Use the Raspberry Pi only as a Chapman-network bridge and file shuttle. Do not run Python training commands on the Pi, and do not submit Slurm from the Pi except through an SSH session on the Chapman login node.

From the machine that has the runner tarball, copy it to the Pi:

```bash
scp neurotwin-a100-runner-<short_sha>.tar.gz \
  <pi_user>@<raspberry_pi_host>:/tmp/
```

From the Pi, copy the bundle to the Chapman login node:

```bash
scp /tmp/neurotwin-a100-runner-<short_sha>.tar.gz \
  <chapman_user>@<chapman_login_host>:~/
```

SSH from the Pi into the Chapman login node before running the operator workflow:

```bash
ssh <chapman_user>@<chapman_login_host>
```

## Environment

Expected baseline:

```text
Python: 3.10
CUDA: 12.1-compatible driver/runtime
PyTorch: >=2.2 with CUDA enabled
Major libraries: MOABB, MNE, MNE-BIDS, NumPy, pandas, PyYAML, SciPy, scikit-learn, tensorboard, wandb
```

Pip fallback when the cluster already provides CUDA-enabled PyTorch:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
python -m pip install -e '.[moabb,cluster]'
python -m pip install -r requirements/cluster-a100.txt
```

## Docker Fallback Details

The packaged helper is the recommended Docker fallback when `conda` or `sbatch` are not available:

```bash
export HOST_GPU_IDS=0,1,2,3,4,5
export GPU_COUNT=6
export NPROC_PER_NODE=6
export CONTAINER_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export A100_CONFIG_TEMPLATE=configs/train/moabb_a100.yaml
export A100_RUN_ID=moabb_a100
export PERSISTENT_ROOT=/raid/scratch/$USER/neurotwin-<short_sha>
bash scripts/run_docker_6gpu.sh "$PERSISTENT_ROOT"
```

For a one-GPU diagnostic, pass one visible host GPU id and force one worker:

```bash
export HOST_GPU_IDS=<host_gpu_id>
export GPU_COUNT=1
export NPROC_PER_NODE=1
export CONTAINER_CUDA_VISIBLE_DEVICES=0
bash scripts/run_docker_6gpu.sh "$PERSISTENT_ROOT"
```

In that diagnostic mode the helper passes Docker `--gpus "\"device=<host_gpu_id>\""` and launches `torchrun --standalone --nproc_per_node=1`. Do not treat a one-GPU diagnostic as the requested 6-GPU run.

For exact Docker flags, environment variables, and agent deployment behavior, use `README_AGENT_DEPLOY.md`. The full Docker run must go through `scripts/run_docker_6gpu.sh` so `docker_run.env` and the current Docker log are produced for the evidence bundle.

The helper delegates the inside-container install, preflight, audit, materialization, training, reporting, and evidence packaging steps to `scripts/docker_a100_inner.sh`. Use `README_AGENT_DEPLOY.md` for the detailed deployment-agent contract; do not copy a raw `docker run` full-run command from this README.

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

## Conda/Slurm A100 Infrastructure Validation

Required resources:

```text
GPU: 1x A100 80GB for the first Slurm validation, or 6x A100 80GB for the Docker/helper heavy lane
CPU: 16 cores
RAM: 128G
Wall time: 02:00:00 for the first infrastructure validation
```

Use a persistent shared root:

```bash
bash scripts/run_full.sh /shared/scratch/$USER/neurotwin
bash scripts/run_full.sh /data/$USER/neurotwin
```

Do not use temporary, relative, checkout-local, or laptop-local roots:

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

## Heavy 6-GPU Slurm Follow-Up

Do not start a long 6-GPU run until local tests, the 1-GPU A100 smoke, and the 3-seed MOABB paper-mode eval pass for this exact commit. If Chapman confirms six A100s are available and `outputs/configs/moabb_a100.materialized.yaml` already exists, the packaged heavy-lane wrapper is:

```bash
export NEUROTWIN_DATA=/path/to/shared/persistent/neurotwin
export RUN_ROOT="$NEUROTWIN_DATA/runs"
export A100_CONFIG_TEMPLATE=configs/train/moabb_a100.yaml
export A100_RUN_ID=moabb_a100
PYTHONPATH=src python3 -m neurotwin.cli cluster materialize-config \
  --template "$A100_CONFIG_TEMPLATE" \
  --prepared-root "$NEUROTWIN_DATA/prepared/moabb_benchmark" \
  --out outputs/configs/moabb_a100.materialized.yaml
RUN_ROOT="$RUN_ROOT" \
sbatch --ntasks-per-node=6 --gres=gpu:a100:6 \
  scripts/slurm/train_a100.sh outputs/configs/moabb_a100.materialized.yaml
```

## Data And Internet

Dataset: MOABB `BNCI2014_001` with `LeftRightImagery`.

Data preparation may need internet if the MOABB cache is not already populated. Training itself must not download data; it reads prepared manifests from the persistent root.

No checkpoints, API keys, private datasets, or raw public neural data are required in the runner.

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
$NEUROTWIN_DATA/gpu_preflight.json
$NEUROTWIN_DATA/docker_run.env
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
$NEUROTWIN_DATA/logs/neurotwin-a100-docker-<generated>.log
```

If the run directory already exists, NeuroTwin reuses the same run id. For a clean rerun, move or archive the previous `$NEUROTWIN_DATA/runs/moabb_a100_smoke` directory first.

## Send Back Evidence

After a run, create the small review bundle from the same extracted runner:

```bash
bash scripts/package_a100_evidence_bundle.sh "$NEUROTWIN_DATA" outputs
```

The evidence zip includes summaries, metrics, tables, figures, prepared manifests/audits, `run/gpu_preflight.json`, `run/docker_run.env`, current-run logs, `COMMIT_HASH.txt`, `README_HANDOFF.md`, `handoff-SHA256SUMS`, and `README_SEND_TO_FRIEND.md`. It excludes `checkpoint*.pt`, raw prepared arrays, runner tarballs, zip artifacts, passwords, API keys, SSH keys, `.env*` files, and private keys.

## Known Limitations

- Docker fallback does not submit Slurm; it runs directly inside the Docker allocation with the GPU list passed to `scripts/run_docker_6gpu.sh`.
- MOABB data preparation may need internet or a populated MOABB cache.
- The default guarded run is configured for 50 smoke steps and `scientific_claim_allowed=false`; the long 6-GPU lane requires `A100_CONFIG_TEMPLATE=configs/train/moabb_a100.yaml`.
- Scientific claims require repeated held-out real-data runs, baseline comparisons, CI-backed reporting, and paper-mode gates.

## Success Condition

The first A100 validation succeeds only when:

- Slurm starts a job on an A100 allocation.
- Or, for the primary Docker path, Docker exposes exactly six CUDA devices and `run/gpu_preflight.json` reports `visible_gpu_count=6`.
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
