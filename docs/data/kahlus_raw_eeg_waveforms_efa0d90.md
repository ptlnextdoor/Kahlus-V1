# Kahlus Raw EEG Waveform Export (`efa0d90`)

This manifest records the out-of-band raw EEG waveform archive prepared for paper figure generation and reproducibility review. The raw archive is intentionally not committed to this repository.

## Artifact

- Archive name: `kahlus-raw-eeg-waveforms-efa0d90.zip`
- Archive format: ZIP, stored with no compression (`zip -0`)
- Size: `79,893,213,364` bytes (`74G` on macOS `ls -lh`)
- ZIP entries: `4,533`
- SHA-256: `52a6dd67c0f27ba0c9184085be6c642de259fa7604e69931e435b83095f1e916`
- Local preparation path: `/Users/krishgarg/Downloads/kahlus-raw-eeg-waveforms-efa0d90.zip`
- Source root: `dgx0-direct:/raid/scratch/kgarg/kahlus/raw/`
- Export date: `2026-07-10`

## Contents

Top-level raw dataset trees inside `kahlus-raw-eeg-waveforms-efa0d90/raw/`:

- `chbmit`
- `eegmmi`
- `siena`
- `sleep-edfx`

The export preserves raw EDF waveform files and companion event/annotation files needed for waveform figures. It is a large raw-data artifact and should be distributed through external artifact storage rather than Git.

## Distribution Status

GitHub cannot store this archive directly as a normal Git object, and this repository does not currently define a large-object storage location for a 74 GB raw ZIP. A distribution link should be added here after the archive is uploaded to an external storage system that supports this size, such as institutional object storage, S3/R2, Google Drive via a native uploader, OSF, Zenodo, or Hugging Face Datasets.

- Download URL: `PENDING_EXTERNAL_STORAGE_UPLOAD`

## Verification

After download, verify the archive before using it:

```bash
shasum -a 256 kahlus-raw-eeg-waveforms-efa0d90.zip
zipinfo -1 kahlus-raw-eeg-waveforms-efa0d90.zip | wc -l
```

Expected values:

- SHA-256: `52a6dd67c0f27ba0c9184085be6c642de259fa7604e69931e435b83095f1e916`
- ZIP entry count: `4533`

## Scope Notes

- This export is for non-clinical paper figure generation and raw waveform inspection.
- Do not describe the artifact as evidence for seizure prediction, diagnosis, treatment, sleep diagnosis, clinical utility, or a brain foundation model claim.
- Keep raw public neural data out of Git history.
