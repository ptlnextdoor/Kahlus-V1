# A100 Runbook

A100 is the canonical cluster target for NeuroTwin v1. H100 configs remain compatible high-memory variants, but public docs and acceptance gates should point here first.

Prepare data before training. Cluster jobs must read local prepared manifests and must not download MOABB, OpenNeuro, or other public data during training.

Local readiness checks:

```bash
PYTHONPATH=src python3 -m neurotwin.cli doctor
PYTHONPATH=src python3 -m neurotwin.cli train --dry-run --config configs/train/neurotwin_v1_a100.yaml
PYTHONPATH=src python3 -m neurotwin.cli estimate --config configs/train/neurotwin_v1_a100.yaml
bash -n scripts/slurm/*.sh
```

Prepare the first real-data benchmark outside the repo:

```bash
scripts/prepare_moabb_benchmark.sh /tmp/neurotwin_moabb_benchmark
```

Submit training:

```bash
sbatch scripts/slurm/train_a100.sh configs/train/moabb_a100.yaml
sbatch scripts/slurm/eval_a100.sh neural_translation_v1 runs/<run_id>
sbatch scripts/slurm/sweep_a100.sh configs/train/moabb_a100.yaml
```

Every real run should write config, split manifest, metrics, best checkpoint, environment, git commit, split hash, report tables, and figure specs. Scientific claims require repeated real-data runs, strict held-out splits, and no synthetic-only or smoke-only labeling.
