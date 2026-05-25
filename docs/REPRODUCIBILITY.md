# Reproducibility

Required for real runs:

- deterministic seed
- config snapshot
- environment capture
- git commit capture
- dataset manifest hash
- split manifest hash
- metrics JSON/JSONL
- prepared event manifest hashes and per-event SHA-256 hashes when `event_manifest.json` is used
- metrics CSV for prepared-manifest training
- checkpoint with model config, task metadata, optimizer state, completed step count, and metrics for prepared-manifest training
- resume path recorded in prepared training metrics when `--resume` is used
- explicit synthetic/demo labels

Core helpers live in `neurotwin.repro`.
