# Remediation Roadmap

No pull request should merge and no expensive A100 job should launch until the P0 protocol audit is complete.

## P0: Evidence-Invalidating

1. Replace forecasting construction with explicit context length, target length, gap, and immutable sample ranges. Require no overlap.
2. Add failing regression tests for the current one-sample-shift defect, then fix both prepared and EEG v1 task builders.
3. Mark every historical affected artifact as `overlapping_target_task`; prohibit reuse of "future-window" wording.
4. Separate model identities: GRU translator, true SSM, Transformer, and experimental NFC. Remove misleading aliases from evidence tables.
5. Replace elementwise inferential intervals with paired subject/patient-level inference.
6. Define a fair baseline/model-selection budget and invalidate unmatched rankings.
7. Make calibration gates evidence-derived; block uncertainty claims until proper scoring and held-out calibration exist.

Stopping condition: local synthetic recovery, deliberate-corruption gates, and a tiny public-data smoke prove the corrected protocol end to end.

## P1: Reproducibility and Benchmark Integrity

1. Add raw content hashes, dataset versions, licenses, preprocessing DAGs, units, reference/filter settings, and train-fitted transform lineage.
2. Reject duplicate record IDs and cross-dataset duplicate content.
3. Make session grouping subject-qualified and require temporal embargo for time splits.
4. Provide an immutable environment/container digest and source-import assertion.
5. Rebuild a raw-to-report CPU smoke package with included fixtures and no hidden paths.
6. Add deterministic distributed tests that keep rank step counts and collectives synchronized.

Stopping condition: an independent clean environment reproduces the smoke report and rejects intentionally corrupted manifests.

## P2: Strong Peer Review

1. Run valid horizon/gap sweeps on BNCI2014-001 and one external dataset.
2. Use tuned persistence, ridge, AR/VAR, TCN, GRU, Transformer, and real SSM baselines; add LaBraM/BIOT only when task-compatible and contamination-audited.
3. Add seed, subject, dataset, and site variance plus failure stratification.
4. Implement coordinate-aware NFC only behind a falsifiable montage/discretization benchmark.
5. Run coordinate, modality, operator, subject-conditioning, and observation-head ablations.
6. Treat masked reconstruction, few-shot adaptation, sleep, and seizure tasks as separate prespecified studies.

Stopping condition: the main advantage survives matched baselines on two datasets with subject-level uncertainty and no nuisance-control explanation.

## P3: Maintainability and Presentation

1. Split giant test/reporting modules along task and evidence boundaries.
2. Remove duplicate aliases, unreachable returns, dead configs, and stale generated files.
3. Generate manuscript tables/figures only from validated matrix rows.
4. Add a terminology glossary distinguishing signal, observation, latent tensor, neural field, source, prediction, reconstruction, and clinical inference.

## Proposed PR Sequence

- PR A, based on audited `main`: task-definition regression tests and historical artifact labels.
- PR B, based on PR A: corrected non-overlapping task builders and gates.
- PR C, based on PR B: subject-level statistics and fair baseline contracts.
- PR D, based on PR C: provenance, deduplication, and locked reproduction package.
- PR E, based on merged P0/P1 main: corrected local real-data smoke.
- PR F, only after E passes: A100 experiment package.
- PR G, after results: NFC coordinate/operator benchmark or explicit retirement of that claim.

Each PR must be reviewed and merged in order. Results from an earlier task definition cannot be compared numerically to later tasks without a visible incompatibility marker.
