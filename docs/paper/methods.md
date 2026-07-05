# Methods

NeuroTwin represents neural recordings as `NeuralEventBatch` objects and splits recording-level manifests before preprocessing or windowing. The experimental NFC path infers a latent neural field with shape `[batch, time, nodes, latent_dim]`, evolves it causally, and compiles it into modality-specific observations. Pair-Operator is retained as an ablation for low-rank relational field updates.

## MOABB ingestion and recording identity

The MOABB adapter ingests public EEG trials through the MOABB paradigm API and converts each trial into a recording-level unit before any benchmark windowing. Each record preserves the dataset identifier, subject, session, run, sampling rate, channel names, source record ID, and adapter provenance. These identifiers are not post hoc annotations: they are the keys used to construct held-out splits and to audit whether any identity group has crossed train, validation, or test.

The prepared real-data artifacts therefore contain manifests rather than raw public EEG arrays. A MOABB trial is first normalized to the NeuroTwin time-by-channel layout, then emitted as a `RecordingRecord` and, when materialized, a `NeuralEventBatch` with the same source record metadata. This keeps the manuscript claim tied to auditable metadata and avoids treating windows from the same source recording as independent split candidates.

## Split construction and leakage checks

`SplitManifest` objects are built from recording-level metadata before preprocessing, augmentation, or windowing. Subject-, session-, site-, and dataset-held-out policies group records by the corresponding identity key, shuffle only the groups, and then assign full groups to train, validation, or test. The leakage audit rejects reused record IDs and overlap of audited keys across train, validation, and test.

Window generation happens after split assignment. Windowed `NeuralEventBatch` objects preserve `source_record_id`, `source_subject_id`, `source_session_id`, and window bounds, so prepared-event audits can trace each window back to exactly one split-manifest record. This is the core leakage-control contract for short-horizon EEG forecasting: training windows cannot be created first and then randomly assigned across identity boundaries.

## Neural Field Compiler path

The NFC implementation is a compiler-style model path rather than a standalone EEG classifier. Source modalities are projected into a latent neural field, optional stimulus and subject-state conditioning are applied, a field-update operator evolves the latent state, and modality-specific observation operators compile the latent field into the requested output modality. For EEG outputs, the current observation operator reads from the latent field into sensor or spectral-proxy outputs.

NFC outputs include the prediction, latent field summaries, projection features, expert-utilization traces, and optional uncertainty maps. The manuscript should interpret these as experimental architecture evidence. NFC and all baselines must be compared under the same split manifests and final held-out reporting rules.

Prepared training uses `train` for optimization, `val` for periodic evaluation and best-checkpoint selection, and `test` only for final held-out reporting. Reports expose `selection_split`, `report_split`, `best_val_mse`, and final `test_*` metrics so checkpoint selection cannot silently use test data.

Real-data v1 artifacts are prepared manifests, not raw public data. MOABB and BIDS adapters write leakage reports, manifest hashes, and benchmark metadata into run artifacts.
