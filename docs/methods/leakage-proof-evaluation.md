# Leakage-Proof Evaluation

NeuroTwin v1 treats split construction as part of the scientific claim.

## Audit contract

The split manifest is built at the recording stage, before preprocessing, augmentation, or window extraction. Each `RecordingRecord` carries stable identity fields and a stable content hash over the manifest metadata. The hash is an audit handle for the record description, not a commitment to raw public data being stored in the repository.

Prepared events must map back to exactly one split-manifest record. When a recording is converted into windows, each window preserves the source record ID, source subject, source session, and window bounds. Evaluation audits use those fields to check that prepared windows did not bypass the original split assignment.

Rules:

- Build `SplitManifest` from recording-level metadata before preprocessing, augmentation, or windowing.
- Use held-out subject, site, and dataset splits as primary tests.
- Run record-ID reuse checks for every split.
- For subject-held-out claims, no subject can appear across train/val/test.
- For session-held-out claims, no session can appear across train/val/test.
- For site-held-out claims, no site can appear across train/val/test.
- For dataset-held-out claims, no dataset can appear across train/val/test.
- Clinical prediction is secondary and cannot rescue a weak Neural Translation result.

Prepared eval audit checks:

- every event file hash listed in `event_manifest.json` is finite and reviewable;
- every prepared event belongs to exactly one split-manifest record;
- every record ID appears in only one split;
- audited identity keys have empty train/val/test overlap for the declared policy;
- prepared event coverage exists for train, validation, and test;
- source-record and window metadata are preserved so repeated or overlapping windows cannot silently cross split boundaries.

Selection and reporting rules:

- `train` is used for optimization only.
- `val` is used for periodic evaluation and best-checkpoint selection.
- `test` is used only for final held-out reporting.
- Reports must expose `selection_split`, `report_split`, best validation metrics, and final test metrics.
- A colocated paper-mode gate proves the artifact contract was checked; it does not by itself allow a scientific claim.
- Scientific claims require real prepared data, a passing leakage audit, strong-baseline comparison, final held-out metrics, and an explicit claim-allowance decision in the run summary.

Main failure modes to catch:

- Subject identity leakage.
- Scanner/site leakage.
- Repeated-session leakage.
- Stimulus or clip leakage.
- Metadata labels that encode the target.
- Training windows generated before split assignment.
- Validation or test metrics used for checkpoint selection.
