# NeuroTwin

NeuroTwin v1 is a research repo for a leakage-proof Neural Translation benchmark and model scaffold. The claim is not "first brain foundation model", "first multimodal brain model", "first stimulus-to-brain model", or a clinical digital twin.

The defensible v1 target is stricter:

> Cross-modal neural translation, missing-modality reconstruction, future-state forecasting, and few-shot subject adaptation under held-out subject/site/dataset splits.

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
- Prepared-manifest NeuroTwin training path with DDP under `torchrun`, checkpoint save/resume, gradient accumulation, bf16 autocast hooks, CSV/JSON metrics, config snapshots, environment info, split hashes, and synthetic/demo labels.
- Prepared-manifest benchmark suite covers the five v1 task families: forecasting, masked reconstruction, cross-modal translation, subject adaptation, and dataset/site generalization.
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
python -m pip install -e '.[bids]'
```

H100 jobs should use prepared local manifests and data roots. Do not depend on internet downloads inside training jobs.

## Local Verification

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m neurotwin.cli doctor
PYTHONPATH=src python3 -m neurotwin.cli data prepare --dataset synthetic --split subject --out-dir /tmp/neurotwin_prepared
PYTHONPATH=src python3 -m neurotwin.cli data smoke --dataset moabb --split subject --out-dir /tmp/neurotwin_moabb_smoke
PYTHONPATH=src python3 -m neurotwin.cli train --dry-run --config configs/train/synthetic_debug.yaml
PYTHONPATH=src python3 -m neurotwin.cli train --config configs/train/prepared_synthetic_debug.yaml --run-root /tmp/neurotwin_runs
PYTHONPATH=src python3 -m neurotwin.cli eval --suite translation_smoke
PYTHONPATH=src python3 -m neurotwin.cli eval audit --suite neural_translation_v1 --event-manifest /tmp/neurotwin_prepared/event_manifest.json --split-manifest /tmp/neurotwin_prepared/split_manifest.json
PYTHONPATH=src python3 -m neurotwin.cli eval --suite neural_translation_v1 --event-manifest /tmp/neurotwin_prepared/event_manifest.json --split-manifest /tmp/neurotwin_prepared/split_manifest.json --train-steps 1
PYTHONPATH=src python3 -m neurotwin.cli report --suite translation_smoke
```

## Repo Rule

Raw public neural data is never committed. The repo stores manifests, configs, adapters, reproducible download/prep scripts, benchmark definitions, and license notes.
