# Remediation Roadmap

No claim-bearing evaluator, training, or expensive A100 pull request should
merge until the P0 protocol audit is complete. This governance freeze may merge
to establish those boundaries.

## C0 HNPH v0.2 Invalidation Freeze

The HNPH v0.2 protocol, machine-readable freeze, canonical implementation
ledger, invalidated result registry, and non-destructive quarantine policy are
now documented. This C0 state invalidates the historical 3.116 overlapping GRU
attribution, the failed seven-A100 rank-drift run, and the completed six-A100 run
whose final gate was false. It does not implement the evaluator or authorize
training.

Raw source caches must be preserved. Derived prepared data, checkpoints, and
evidence associated with invalid runs must be quarantined and excluded from
canonical discovery. Expensive training remains blocked until a clean rebuild
passes source-hash, zero-person-overlap, physical-unit/rate/channel, no-silent-
truncation, baseline-completeness, distributed-lockstep, and independent
reproduction gates.

## Delivery State

- R0 forensic audit merged in `564d80f2`.
- R1 forecast/provenance contract merged in `06654d27`.
- R2 non-overlapping builders merged in PR #56. They are a necessary generic
  forecast construction repair, not HNPH evaluator, baseline, or training
  authorization; historical shifted-window builders remain readable only for
  compatibility and emit `kahlus.forecast.v1_overlap` as ineligible metadata.
- `p0_remediation_complete=false` remains in effect. No A100 work is allowed
  until R1 through R6 are merged and their local protocol gates pass.

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
