# A100 Runbook

A100 is the canonical cluster target for NeuroTwin v1. H100 configs remain compatible high-memory variants, but public docs and acceptance gates should point here first.

Prepare data before training. Cluster jobs must read local prepared manifests and must not download MOABB, OpenNeuro, or other public data during training.
Set `NEUROTWIN_DATA` to a persistent shared filesystem location; prepared benchmark artifacts belong under `$NEUROTWIN_DATA/prepared/`, not node-local `/tmp`.

For the first Chapman run, prefer the guarded one-command path:

```bash
scripts/cluster/chapman_a100_first_run.sh /path/to/shared/persistent/neurotwin
```

That launcher prepares MOABB, verifies `window_count=18144`, materializes absolute manifest paths, dry-runs, and submits exactly one A100 smoke job.

Local readiness checks:

```bash
PYTHONPATH=src python3 -m neurotwin.cli doctor
PYTHONPATH=src python3 -m neurotwin.cli train --dry-run --config configs/train/neurotwin_v1_a100.yaml
PYTHONPATH=src python3 -m neurotwin.cli estimate --config configs/train/neurotwin_v1_a100.yaml
bash -n scripts/slurm/*.sh
```

Prepare the first real-data benchmark outside the repo:

```bash
export NEUROTWIN_DATA=/path/to/persistent/neurotwin
scripts/prepare_moabb_benchmark.sh
```

Do not submit A100 jobs unless the benchmark preparation prints `eval_audit_passed=True`, `window_count > 0`, and nonzero train/val/test entries in `window_counts_by_split`.
For BNCI2014_001, the locked MOABB benchmark defaults to `window_length=128` and `stride=128`; larger windows can produce zero runnable windows.

Submit training:

```bash
export RUN_ROOT=/path/to/shared/persistent/neurotwin/runs
PYTHONPATH=src python3 -m neurotwin.cli cluster preflight \
  --config configs/train/moabb_a100_chapman.yaml \
  --run-root "$RUN_ROOT" \
  --require-cuda \
  --require-prepared-windows
sbatch scripts/slurm/train_a100.sh configs/train/moabb_a100_chapman.yaml
sbatch scripts/slurm/eval_a100.sh "$RUN_ROOT/<run_id>"
sbatch scripts/slurm/sweep_a100.sh configs/train/moabb_a100_chapman_seed*.yaml
```

Every real run should write config, split manifest, metrics, best checkpoint, environment, git commit, split hash, report tables, and figure specs. Scientific claims require repeated real-data runs, strict held-out splits, and no synthetic-only or smoke-only labeling.
