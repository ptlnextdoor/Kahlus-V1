# A100 Runbook

A100 is the canonical cluster target for NeuroTwin v1. H100 configs remain compatible high-memory variants, but public docs and acceptance gates should point here first.

Prepare data before training. Cluster jobs must read local prepared manifests and must not download MOABB, OpenNeuro, or other public data during training.
Set `NEUROTWIN_DATA` to a persistent shared filesystem location; prepared benchmark artifacts belong under `$NEUROTWIN_DATA/prepared/`, not node-local `/tmp`.

For the first Chapman run, prefer the guarded one-command path:

```bash
bash scripts/run_full.sh /path/to/shared/persistent/neurotwin
```

That launcher prepares MOABB, verifies `window_count=18144`, materializes absolute manifest paths under `outputs/configs/`, dry-runs, and submits exactly one A100 smoke job. `scripts/cluster/chapman_a100_first_run.sh` is a compatibility wrapper around the same path.

## Fast Iteration Lane

Use one A100 for short validation runs before spending a multi-day allocation:

```bash
PYTHONPATH=src python3 -m unittest
git diff --check
bash scripts/run_smoke.sh outputs/smoke-head
bash scripts/run_full.sh /path/to/shared/persistent/neurotwin
```

Then run the 3-seed MOABB paper-mode gate on prepared manifests:

```bash
export NEUROTWIN_DATA=/path/to/shared/persistent/neurotwin
export PREPARED_DIR="$NEUROTWIN_DATA/prepared/moabb_benchmark"
export EVAL_DIR="$NEUROTWIN_DATA/eval/moabb_3seed_head"
python3 -m neurotwin.cli eval \
  --suite neural_translation_v1 \
  --paper-mode \
  --seeds 0 1 2 \
  --event-manifest "$PREPARED_DIR/event_manifest.json" \
  --split-manifest "$PREPARED_DIR/split_manifest.json" \
  --window-length 128 \
  --stride 128 \
  --train-steps 3 \
  --out-dir "$EVAL_DIR"
```

MOABB EEG is expected to skip `tribe_style`; this gate validates leakage audits, baseline reporting, seed aggregation, and the paper artifact contract. It does not set `scientific_claim_allowed=true`; that remains an explicit run-summary decision.

## Heavy 6-GPU Lane

Start one 6x A100 80GB run only after local tests, the 1-GPU smoke, and the 3-seed MOABB gate pass for the exact committed artifact. Do not assume a seventh GPU is available unless Slurm confirms it.
When `A100_RUN_ID` is not `moabb_a100_smoke`, the guarded Docker and Slurm helpers default to running that paper-mode gate before long training. After training they copy the small paper-mode artifacts into the run directory, write the run report, run leakage-demo and identity-probe diagnostics, and generate `EEG_MODEL_CARD.md`.

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

Use short 1-GPU jobs for debugging. Do not retry failed multi-GPU runs blindly; inspect logs, metrics, checkpoints, and manifests first.

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
  --config outputs/configs/moabb_a100.materialized.yaml \
  --run-root "$RUN_ROOT" \
  --require-cuda \
  --require-prepared-windows \
  --expect-window-count 18144 \
  --expect-split-windows train:12096,val:2016,test:4032
sbatch scripts/slurm/train_a100.sh outputs/configs/moabb_a100.materialized.yaml
sbatch scripts/slurm/eval_a100.sh "$RUN_ROOT/<run_id>"
sbatch scripts/slurm/sweep_a100.sh outputs/configs/moabb_a100_seed*.yaml
```

Every real run should write config, split manifest, metrics, best checkpoint, environment, git commit, split hash, report tables, figure specs, `LEAKAGE_AUDIT.json`, `CLAIM_GATE.json`, baseline rankings, seed aggregates, leakage-demo output, identity-probe output, and `EEG_MODEL_CARD.md`. A passed paper-mode gate means the artifact contract is satisfied; scientific/model claim allowance is controlled by `summary.json` and stays false unless that summary explicitly allows it. The allowed evidence statement is leakage-proof evaluation and infrastructure validation, not model superiority.
