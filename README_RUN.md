# NeuroTwin A100 Run Bundle

This bundle runs one controlled NeuroTwin MOABB infrastructure validation on an A100 cluster. It verifies imports, data preparation, leakage/window gates, CUDA visibility, prepared-manifest training, checkpoint writing, and report generation. It is not a scientific result and not the 3-seed acceptance run.

## What The Run Does

The smoke command validates local package, CLI, synthetic data, prepared training, and reporting. The full command validates the real MOABB preparation path and launches one A100 infrastructure job after exact window-count and cluster preflight gates pass.

## Repo And Commit

```bash
git clone https://github.com/ptlnextdoor/Kahlus-V1.git
cd Kahlus-V1
git checkout <COMMIT_HASH_FROM_HANDOFF>
git rev-parse HEAD
```

Use the exact commit hash supplied in the handoff message. Do not run from an edited worktree for the first A100 validation.

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

`scripts/run_full.sh` refuses `/tmp`, `/private/tmp`, `/var/tmp`, missing roots, unwritable `logs/`, missing manifests after preparation, bad window counts, and failed preflight. It materializes:

```text
outputs/configs/moabb_a100.materialized.yaml
```

with absolute manifest paths under:

```text
/path/to/shared/persistent/neurotwin/prepared/moabb_benchmark/
```

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
logs/neurotwin-a100-full-<jobid>.out
logs/neurotwin-a100-full-<jobid>.err
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
