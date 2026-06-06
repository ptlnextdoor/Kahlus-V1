# NeuroTwin

NeuroTwin v1 is a research repo for a leakage-proof Neural Translation benchmark and model scaffold. The current experimental architecture is NeuroTwin NFC, a Neural Field Compiler. The claim is not "first brain foundation model", "first multimodal brain model", "first stimulus-to-brain model", or a clinical digital twin.

The defensible v1 target is stricter:

> Cross-modal neural translation, missing-modality reconstruction, future-state forecasting, and few-shot subject adaptation under held-out subject/site/dataset splits.

NFC treats fMRI, EEG, behavior, stimulus responses, and future modalities as partial observations of one latent neural field. Pair-Operator remains usable, but only as a baseline/ablation for low-rank relational field updates.

For the current repo state and NFC pivot map, see `docs/research/neurotwin_project_state.md`. Track A is reproducibility and claim-gate evidence. Track B is the NFC model path. The next A100 step is strict 1x NFC synthetic diagnostic only, not Algonauts or 6x DDP.

Primary competitors are explicit first-class baselines: TRIBE v2, BrainVista, Brain-OF, BrainOmni, Brain Harmony, plus Transformer, SSM/Mamba, and modality-specialist baselines. Brain-OF is treated as the main benchmark opponent for generic multimodal neural foundation modeling.

## Current Scaffold

- Shared `NeuralEventBatch` schema for fMRI, EEG, MEG, spikes, behavior, stimulus, anatomy, and clinical metadata.
- `SplitManifest` built at recording-manifest time, before preprocessing, windowing, or augmentation.
- Leakage checks and audits for record reuse, held-out group overlap, repeated windows, and forbidden target metadata.
- Required v1 Neural Translation task registry.
- Competitor registry that names crowded lanes instead of pretending they do not exist.
- Synthetic CPU smoke data for testable benchmark plumbing. These results are not science.
- Optional MOABB EEG adapter and BIDS/OpenNeuro manifest scanner. BIDS can emit prepared events from precomputed `_timeseries.npy/.npz/.tsv/.csv` derivatives without raw NIfTI preprocessing.
- Prepared `event_manifest.json` artifacts for offline train/eval jobs. Synthetic and MOABB preparation emit `NeuralEventBatch` `.npz` files; BIDS is manifest-only in this pass.
- Prepared-manifest NeuroTwin training path with single-task or `neural_translation_v1` multi-task training, DDP under `torchrun`, checkpoint save/resume, best checkpoints, objective-level JSONL metrics, gradient accumulation, bf16 autocast hooks, CSV/JSON metrics, config snapshots, environment info, split hashes, and synthetic/demo labels.
- Prepared training uses train for optimization, val for periodic evaluation and best-checkpoint selection, and test for final held-out reporting.
- Prepared-manifest benchmark suite covers the five v1 task families: forecasting, masked reconstruction, cross-modal translation, subject adaptation, and dataset/site generalization. Baseline failures are explicit and excluded from rankings.
- A100 is the canonical cluster target through `configs/train/*_a100.yaml`, `scripts/slurm/*_a100.sh`, and `docs/A100_RUNBOOK.md`; H100 remains a compatible high-memory variant.
- Chapman A100 first launch has a guarded one-command path in `scripts/cluster/chapman_a100_first_run.sh`, with explicit preflight against placeholder paths, zero-window data, missing CUDA, and non-persistent run roots.
- Modular NeuroTwin model internals now expose modality encoders, `transformer`/`ssm_fallback` backbone selection, geometry/metadata encoders, projection heads, and leakage-safe subject-adapter controls. `mamba` remains an upstream baseline target, not a wired NeuroTwin backbone selector.
- Experimental NFC internals expose `NeuralFieldCompiler`, `LatentNeuralField`, causal field updates, low-rank pair kernels, observation operators, stimulus conditioning, uncertainty maps, and a synthetic latent-field suite. Synthetic NFC outputs are plumbing checks, not science.
- Reproducibility helpers for deterministic seeds, config snapshots, environment capture, git commit capture, and stable manifest hashes.
- CLI surface:
  - `nt doctor`
- `nt data prepare --dataset synthetic --split subject`
- `nt data smoke --dataset moabb --split subject --out-dir /tmp/neurotwin_moabb_smoke`
- `nt data audit --dataset synthetic`
  - `nt split audit --dataset synthetic --split subject`
  - `nt estimate --config <experiment>`
  - `nt train --dry-run --config <experiment>`
  - `nt train --config <experiment>`
  - `nt train --config <experiment> --resume runs/<run_id>/checkpoint.pt`
  - `nt eval --suite translation_smoke`
  - `nt eval --suite neural_translation_v1 --event-manifest <prepared>/event_manifest.json --split-manifest <prepared>/split_manifest.json`
  - `nt eval audit --suite neural_translation_v1 --event-manifest <prepared>/event_manifest.json --split-manifest <prepared>/split_manifest.json`
  - `nt report --suite translation_smoke`
  - `nt report --run-dir runs/<run_id>`
  - `nt report --compare runs/<run_a> runs/<run_b> --out-dir reports/compare`

## Install

Local CPU/dev:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Optional real-data adapters:

```bash
python -m pip install -e '.[moabb]'
```

BIDS derivative scanning is included in the base install; it does not install
heavy BIDS preprocessing libraries.

A100/H100 jobs should use prepared local manifests and data roots. Do not depend on internet downloads inside training jobs.

For the first Chapman A100 run:

```bash
scripts/cluster/chapman_a100_first_run.sh /path/to/shared/persistent/neurotwin
```

## Local Verification

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m neurotwin.cli doctor
PYTHONPATH=src python3 -m neurotwin.cli data prepare --dataset synthetic --split subject --out-dir /tmp/neurotwin_prepared
PYTHONPATH=src python3 -m neurotwin.cli data smoke --dataset moabb --split subject --out-dir /tmp/neurotwin_moabb_smoke
scripts/prepare_moabb_smoke.sh /tmp/neurotwin_moabb_smoke
scripts/prepare_moabb_benchmark.sh /tmp/neurotwin_moabb_benchmark
PYTHONPATH=src python3 -m neurotwin.cli train --dry-run --config configs/train/neurotwin_v1_a100.yaml
PYTHONPATH=src python3 -m neurotwin.cli train --dry-run --config configs/train/synthetic_debug.yaml
PYTHONPATH=src python3 -m neurotwin.cli train --config configs/train/prepared_synthetic_debug.yaml --run-root /tmp/neurotwin_runs
PYTHONPATH=src python3 -m neurotwin.cli train --config configs/train/moabb_smoke_locked.yaml --run-root /tmp/neurotwin_moabb_runs
PYTHONPATH=src python3 -m neurotwin.cli eval --suite translation_smoke
PYTHONPATH=src python3 -m neurotwin.cli eval audit --suite neural_translation_v1 --event-manifest /tmp/neurotwin_prepared/event_manifest.json --split-manifest /tmp/neurotwin_prepared/split_manifest.json
PYTHONPATH=src python3 -m neurotwin.cli eval --suite neural_translation_v1 --event-manifest /tmp/neurotwin_prepared/event_manifest.json --split-manifest /tmp/neurotwin_prepared/split_manifest.json --train-steps 1
PYTHONPATH=src python3 -m neurotwin.cli eval --suite nfc_synthetic --out-dir /tmp/neurotwin_nfc_synthetic --train-steps 1 --seed 0
PYTHONPATH=src python3 -m neurotwin.cli report --suite translation_smoke
bash -n scripts/slurm/*.sh
```

## Repo Rule

Raw public neural data is never committed. The repo stores manifests, configs, adapters, reproducible download/prep scripts, benchmark definitions, and license notes.
