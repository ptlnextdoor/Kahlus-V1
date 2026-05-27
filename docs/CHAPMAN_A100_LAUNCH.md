# Chapman A100 Controlled Launch

This is the operational command sheet for the first controlled NeuroTwin MOABB run on the Chapman A100 cluster. This is a launch-readiness smoke/benchmark run, not a scientific result.

Do not submit an A100 job until MOABB benchmark preparation passes the window gate with nonzero train/val/test windows.

## Local Status

The current local workspace is a Mac environment, not Chapman:

- Host checked: `Aayushyas-MacBook-Pro-2.local`
- OS checked: macOS/Darwin ARM64
- `sbatch`: unavailable locally
- `squeue`: unavailable locally
- CUDA locally: `False`
- CUDA device count locally: `0`
- Local `NEUROTWIN_DATA`, `MOABB_DATA`, `BIDS_ROOT`, `RUN_ROOT`: unset

The A100 launch must be run from a Chapman shell with SLURM and A100 access.

For the lowest-resistance path, use the guarded launcher:

```bash
scripts/cluster/chapman_a100_first_run.sh /path/to/shared/persistent/neurotwin
```

It performs the setup, data preparation, window gate, absolute-path config generation, dry-run, and one-job submit steps below.

## Cluster Setup

On Chapman:

```bash
cd /path/to/Kahlus\ Vidya\ v1
mkdir -p logs

export NEUROTWIN_DATA=/path/to/shared/persistent/neurotwin
export MOABB_DATA=$NEUROTWIN_DATA/moabb
export BIDS_ROOT=$NEUROTWIN_DATA/bids
export RUN_ROOT=$NEUROTWIN_DATA/runs

mkdir -p "$NEUROTWIN_DATA" "$MOABB_DATA" "$BIDS_ROOT" "$RUN_ROOT"
```

Use a persistent shared filesystem for `NEUROTWIN_DATA`. Do not use node-local `/tmp` for cluster artifacts.

## Prepare MOABB Benchmark Data

```bash
bash scripts/prepare_moabb_benchmark.sh
```

Required gate before launch:

```txt
eval_audit_passed=True
window_count=18144
window_counts_by_split=train:12096,val:2016,test:4032
```

Expected completed task statuses for this EEG-only MOABB path:

```txt
summary_task_status_future_state_forecasting=completed
summary_task_status_masked_neural_reconstruction=completed
summary_task_status_few_shot_subject_adaptation=completed
```

Expected skipped tasks:

```txt
cross_modal_translation: need paired train/test windows for two modalities
dataset_site_generalization: need train/test windows from different datasets or sites
```

## Create Chapman Config

The config loader does not expand environment variables inside YAML. Do not write literal `$NEUROTWIN_DATA` into the config.

Create a Chapman-specific config with absolute manifest paths:

```bash
cp configs/train/moabb_a100.yaml configs/train/moabb_a100_chapman.yaml

python3 - <<'PY'
import os
from pathlib import Path

cfg = Path("configs/train/moabb_a100_chapman.yaml")
root = Path(os.environ["NEUROTWIN_DATA"]).resolve()
text = cfg.read_text(encoding="utf-8")
text = text.replace(
    "event_manifest: /path/to/moabb_prepared/event_manifest.json",
    f"event_manifest: {root}/prepared/moabb_benchmark/event_manifest.json",
)
text = text.replace(
    "split_manifest: /path/to/moabb_prepared/split_manifest.json",
    f"split_manifest: {root}/prepared/moabb_benchmark/split_manifest.json",
)
cfg.write_text(text, encoding="utf-8")
PY
```

Confirm the config contains absolute paths:

```bash
grep -A3 '^data:' configs/train/moabb_a100_chapman.yaml
```

Expected shape:

```yaml
data:
  event_manifest: /path/to/shared/persistent/neurotwin/prepared/moabb_benchmark/event_manifest.json
  split_manifest: /path/to/shared/persistent/neurotwin/prepared/moabb_benchmark/split_manifest.json
```

## Dry Run

Run these in the cluster environment:

```bash
PYTHONPATH=src python3 -m neurotwin.cli doctor
PYTHONPATH=src python3 -m neurotwin.cli cluster preflight \
  --config configs/train/moabb_a100_chapman.yaml \
  --run-root "$RUN_ROOT" \
  --require-cuda \
  --require-prepared-windows
PYTHONPATH=src python3 -m neurotwin.cli train --dry-run --config configs/train/moabb_a100_chapman.yaml
```

Required checks:

- `doctor` runs successfully.
- CUDA is available in the SLURM/A100 context.
- Config hash prints.
- Dry run estimates model/runtime size.
- Manifest paths are absolute and point under `$NEUROTWIN_DATA/prepared/moabb_benchmark`.

## Submit One A100 Run

Submit exactly one controlled run first:

```bash
RUN_ROOT=$NEUROTWIN_DATA/runs sbatch scripts/slurm/train_a100.sh configs/train/moabb_a100_chapman.yaml
```

Watch:

```bash
squeue -u $USER
tail -f logs/*.out
```

## First-Launch Success Criteria

This first A100 launch counts as successful only if:

- Job starts on an A100 allocation.
- `nt doctor` in the job reports CUDA available and device count greater than zero.
- Dry run succeeds before `torchrun`.
- Prepared event and split manifests are found.
- Training sees nonzero train/val/test prepared windows.
- Checkpoints save under `$NEUROTWIN_DATA/runs`.
- Metrics/report artifacts write without rank collisions.

## Not A Scientific Result Yet

Do not claim:

- NeuroTwin is better.
- The paper is ready.
- This is a scientific result.

After one launch succeeds, run 3 seeds, produce baseline reports, and use `nt report --compare` before interpreting results.
