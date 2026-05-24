# H100 Runbook

Prepare data before training. H100 jobs must not require internet access.

Local dry run:

```bash
PYTHONPATH=src python3 -m neurotwin.cli doctor
PYTHONPATH=src python3 -m neurotwin.cli data prepare --dataset synthetic --split subject --out-dir /tmp/neurotwin_prepared
PYTHONPATH=src python3 -m neurotwin.cli data smoke --dataset moabb --split subject --out-dir /tmp/neurotwin_moabb_smoke
PYTHONPATH=src python3 -m neurotwin.cli eval audit --suite neural_translation_v1 \
  --event-manifest /tmp/neurotwin_prepared/event_manifest.json \
  --split-manifest /tmp/neurotwin_prepared/split_manifest.json
PYTHONPATH=src python3 -m neurotwin.cli eval --suite neural_translation_v1 \
  --event-manifest /tmp/neurotwin_prepared/event_manifest.json \
  --split-manifest /tmp/neurotwin_prepared/split_manifest.json \
  --train-steps 1
PYTHONPATH=src python3 -m neurotwin.cli train --config configs/train/prepared_synthetic_debug.yaml --run-root /tmp/neurotwin_runs
PYTHONPATH=src python3 -m neurotwin.cli train --config configs/train/prepared_synthetic_debug.yaml \
  --run-root /tmp/neurotwin_runs \
  --resume /tmp/neurotwin_runs/prepared_synthetic_debug/checkpoint.pt
PYTHONPATH=src python3 -m neurotwin.cli train --dry-run --config configs/train/neurotwin_v1_h100.yaml
PYTHONPATH=src torchrun --standalone --nproc_per_node=2 \
  -m neurotwin.cli train --config configs/train/prepared_synthetic_debug.yaml --run-root /tmp/neurotwin_ddp_runs
```

SLURM:

```bash
sbatch scripts/slurm/train_h100.sh configs/train/neurotwin_v1_h100.yaml
sbatch scripts/slurm/eval_h100.sh neural_translation_v1 runs/<run_id>
sbatch scripts/slurm/sweep_h100.sh
```

Every real run should write a config copy, split manifest, metrics, checkpoints, environment info, git commit, and result summary.
Prepared training also writes `metrics.csv`, stores optimizer state in `checkpoint.pt`, honors `gradient_accumulation_steps`, and enables bf16 autocast when `precision: bf16` is configured.
When launched through `torchrun`, prepared training initializes `torch.distributed`, wraps the model in DDP, writes rank-specific JSONL metrics, and leaves shared checkpoints/reports to rank zero.

For real cluster runs, edit `configs/train/neurotwin_v1_h100.yaml` so `data.event_manifest` and `data.split_manifest` point to prepared local files. Training jobs must not download MOABB/OpenNeuro/other data.
