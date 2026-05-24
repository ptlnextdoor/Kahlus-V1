# Leakage

Primary split claims require matching audits:

- subject-held-out: no subject overlap
- session-held-out: no session overlap
- site-held-out: no site overlap
- dataset-held-out: no dataset overlap
- run-held-out: no run overlap when run metadata exists
- window-held-out: no repeated or overlapping source windows across splits

Metadata fields that directly encode targets are forbidden in benchmark features unless an experiment explicitly declares a metadata-probe ablation.

Useful commands:

```bash
PYTHONPATH=src python3 -m neurotwin.cli split audit --dataset synthetic --split subject
PYTHONPATH=src python3 -m neurotwin.cli data audit --dataset synthetic
PYTHONPATH=src python3 -m neurotwin.cli eval audit --suite neural_translation_v1 \
  --event-manifest /tmp/neurotwin_prepared/event_manifest.json \
  --split-manifest /tmp/neurotwin_prepared/split_manifest.json
```

Prepared eval audit checks:

- event file SHA-256 hashes from `event_manifest.json`
- every prepared event maps to exactly one split-manifest record
- split policy leakage through `audit_split_manifest`
- train/val/test prepared event coverage
- repeated prepared windows across split boundaries via preserved `source_record_id`
