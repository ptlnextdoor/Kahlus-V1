# Historical MOABB Window Diagnostic

## Purpose

This local-only diagnostic answers the mentor question: why can a ridge model
perform well on the recovered BNCI2014-001 task? It is not a new benchmark
result, a figure source for the paper's claimed model result, or a replacement
for the missing historical checkpoint.

## What Was Verified

The MOABB adapter now requests MNE epochs so it retains the actual epoch
sampling rate and channel names. MOABB's documented array pipeline multiplies
MNE epoch values by `dataset.unit_factor` (normally `1e6`); the adapter applies
the same conversion explicitly and records `signal_unit: uV`.

On the local public BNCI2014-001 cache, an export over subjects 1, 2, and 3
produced 22-channel, 250 Hz windows. The historical configuration used a
127-sample context and a one-sample target offset. Consequently, each target
shares 126 of its 127 samples with its input. The export manifest marks this as
`kahlus.forecast.v1_overlap` and `claim_eligible: false`.

The small local export includes real waveform snippets, ridge predictions,
persistence predictions, per-channel lag correlations, recording IDs, subject
IDs, and start samples. It explicitly does **not** include a Kahlus prediction:
the repository and recovered evidence release do not contain a
provenance-matched historical checkpoint or prediction artifact. A newly trained
local model must not be labeled as the source of the recovered 3.116 result.

## Reproduction

The command writes public waveform snippets only under the chosen output path;
it never commits raw data to the repository. Set the cache paths to existing
locations, or allow MOABB to populate a persistent local cache.

```bash
PYTHONPATH=src python3 scripts/export_moabb_historical_window_evidence.py \
  --out-dir /tmp/kahlus-moabb-window-evidence \
  --subjects 1 2 3 \
  --max-trials 60 \
  --mne-data "$MNE_DATA" \
  --bnci-data-path "$MNE_DATASETS_BNCI_PATH"

PYTHONPATH=src python3 scripts/analysis/plot_ridge_eeg_diagnostics.py \
  --npz /tmp/kahlus-moabb-window-evidence/moabb_historical_window_evidence.npz \
  --out /tmp/kahlus-moabb-window-evidence/figures \
  --max-display-channels 4 \
  --mentor-two-figures
```

## Disposition Of The Last A100 Bundle

`kahlus-multidataset-a100-sendable-6x-efa0d90-stability-v2.zip` is preserved as
failure/provisional evidence only. Its archive contains no waveform arrays; its
own `summary.json` reports `scientific_claim_allowed: false`, while its selected
metric and held-out aggregate test metric are not comparable. It must not supply
paper figures, paper metrics, or an Amrith update.

## Next Evidence Needed

To add a Kahlus trace to the second diagnostic figure, obtain either the exact
historical checkpoint plus architecture/configuration and split identity, or a
prediction export with matching `test_record_ids`, start samples, channel order,
sampling rate, and unit. Until then, ridge versus persistence and the explicit
overlap visualization are the honest diagnostic answer.
