# Data

Raw public neural data is never committed.

Raw EEG waveform export manifests:

- `docs/data/kahlus_raw_eeg_waveforms_efa0d90.md`: out-of-band 74 GB raw EDF ZIP manifest for paper figure generation.

Supported paths in this pass:

- `synthetic`: deterministic CPU plumbing tests.
- `moabb`: optional EEG adapter. Requires `pip install -e '.[moabb]'` and dataset preparation before training. The locked real-data smoke uses `BNCI2014_001`, `LeftRightImagery`, a small subject/trial subset, and a subject-held-out split.
- `moabb benchmark`: `scripts/prepare_moabb_benchmark.sh` prepares the first locked real benchmark protocol with subject-held-out splits and local manifests for A100/H100 jobs. It should run before cluster training; cluster jobs must not download data.
- `bids`: derivative-only manifest scanner for BIDS/OpenNeuro-style datasets. It parses filenames plus `participants.tsv`, `events.tsv`, and `scans.tsv`; it does not perform heavy neuroimaging preprocessing.

Prepared output contract:

- `split_manifest.json`: recording-level split built before preprocessing/windowing.
- `data_manifest.json`: recording metadata and raw/local paths.
- `leakage_report.json`: split leakage status.
- `event_manifest.json`: optional prepared `NeuralEventBatch` artifacts for datasets where arrays are available. Synthetic and MOABB emit this; BIDS emits it only when precomputed time-series derivatives are present.
- `events/*.npz`: prepared arrays with SHA-256 hashes recorded in `event_manifest.json`.

BIDS derivative support:

- Sidecar names next to the source signal: `<bids_stem>_timeseries.npy`, `.npz`, `.tsv`, or `.csv`.
- Or under `derivatives/neurotwin/<same_relative_parent>/<bids_stem>_timeseries.*`.
- Arrays must already be preprocessed and shaped `[time, region_or_channel]`.
- `.npz` files may use a `signal` array and optional `labels` array.
- Derivatives must be finite 2D arrays with at least two timepoints; optional labels must match the space axis.
- This repo intentionally does not implement heavy fMRI preprocessing from raw NIfTI.

Example:

```bash
PYTHONPATH=src python3 -m neurotwin.cli data prepare --dataset synthetic --split subject --out-dir /tmp/neurotwin_prepared
PYTHONPATH=src python3 -m neurotwin.cli eval audit --suite neural_translation_v1 \
  --event-manifest /tmp/neurotwin_prepared/event_manifest.json \
  --split-manifest /tmp/neurotwin_prepared/split_manifest.json
PYTHONPATH=src python3 -m neurotwin.cli eval --suite neural_translation_v1 \
  --event-manifest /tmp/neurotwin_prepared/event_manifest.json \
  --split-manifest /tmp/neurotwin_prepared/split_manifest.json \
  --train-steps 1
scripts/prepare_moabb_smoke.sh /tmp/neurotwin_moabb_smoke
scripts/prepare_moabb_benchmark.sh /tmp/neurotwin_moabb_benchmark
```

Training jobs should read prepared manifests and local data roots. Internet downloads do not happen inside H100 jobs.
