# Chapman A100 Quickstart

This is the shortest safe path for the first NeuroTwin MOABB A100 launch. It proves infrastructure only: CUDA, manifests, nonzero windows, checkpoints, and reports. It is not a scientific result.

## One Command

From a Chapman shell on the PR branch:

```bash
scripts/cluster/chapman_a100_first_run.sh /path/to/shared/persistent/neurotwin
```

The path must be an absolute shared filesystem path, not node-local `/tmp`.

The launcher will:

- create `logs/`, `NEUROTWIN_DATA`, `MOABB_DATA`, `BIDS_ROOT`, and `RUN_ROOT`;
- prepare MOABB benchmark artifacts under `$NEUROTWIN_DATA/prepared/moabb_benchmark`;
- require `eval_audit_passed=True`;
- require `window_count=18144`;
- require `window_counts_by_split=train:12096,val:2016,test:4032`;
- materialize `configs/train/moabb_a100_chapman.yaml` with absolute manifest paths;
- run doctor, preflight, and training dry-run;
- submit exactly one A100 job.

## Manual Preflight

If submitting manually, do not use placeholder configs. Generate or edit a config with absolute manifest paths, then run:

```bash
PYTHONPATH=src python3 -m neurotwin.cli cluster preflight \
  --config configs/train/moabb_a100_chapman.yaml \
  --run-root /path/to/shared/persistent/neurotwin/runs \
  --require-cuda \
  --require-prepared-windows
```

Required output includes:

```txt
preflight_passed=True
cuda_available=True
window_count=18144
window_counts_by_split=train:12096,val:2016,test:4032
```

## Submit Manually

```bash
export RUN_ROOT=/path/to/shared/persistent/neurotwin/runs
sbatch scripts/slurm/train_a100.sh configs/train/moabb_a100_chapman.yaml
```

`train_a100.sh` requires an explicit config and persistent absolute `RUN_ROOT`. It refuses placeholder/default launches.

## Success Boundary

Success means the first A100 job starts, sees CUDA, finds prepared manifests, trains on nonzero train/val/test windows, writes checkpoints under `$RUN_ROOT`, and produces report artifacts.

Do not claim model superiority or paper readiness from this run. Scientific interpretation requires at least 3 real MOABB seeds, baselines, and `nt report --compare`.
