# HNPH Data Migration and Quarantine Policy

**Applies to:** `kahlus.hnph.phase0.v0.2`
**Effective date:** 2026-07-10

The invalid runs do not justify deleting source data. They do require a hard
boundary between retained raw bytes and contaminated derived artifacts.

## Classification

| Class | Required disposition | Allowed use |
| --- | --- | --- |
| Official raw source files and download caches | Preserve unchanged; make read-only where feasible; inventory and hash | Clean canonical rebuild after provenance verification |
| Source metadata adjacent to raw files | Preserve; validate against official source and raw hashes | Provenance reconstruction only until verified |
| Prepared arrays, windows, anchors, split manifests, fitted transforms, and derived caches used by invalid runs | Quarantine | Forensic inspection only |
| Checkpoints, optimizer state, calibrators, and selection state from invalid runs | Quarantine | Forensic inspection only; no warm start |
| Predictions, metrics, reports, figures, tables, logs, and evidence bundles from invalid runs | Quarantine | Invalidation audit only |
| New v0.2 prepared and evidence artifacts | Build in a fresh canonical namespace after all migration gates pass | Uses declared by the frozen protocol |

No raw public neural data may be committed to git. This policy does not name or
assume filesystem locations that have not been verified by an inventory.

## Non-Destructive Migration Procedure

1. Stop all canonical loaders from discovering legacy derived artifacts.
2. Inventory artifacts by content hash, size, owner/run association, and data
   class before moving or changing permissions.
3. Mark artifacts associated with `INV-001`, `INV-002`, or `INV-003` as
   `quarantined_invalid`; do not infer validity from a successful process exit.
4. Preserve raw caches in place when safe. Never move raw data into git or an
   evidence bundle.
5. Move or access-control only derived prepared data, checkpoints, and evidence
   so canonical discovery cannot load them. Record old and new locations in an
   operator migration manifest; this document intentionally invents no paths.
6. Recompute SHA-256 after any copy or move and retain the pre-migration hash.
7. Build v0.2 data from verified raw sources into a new namespace. Do not copy
   prepared arrays, split assignments, fitted transforms, checkpoints,
   calibrators, or baseline rankings from quarantine.
8. Require an independent audit before any legacy item leaves quarantine.

Deletion is not part of C0. Later deletion requires explicit operator approval,
a path-scope check, a completed inventory, and confirmation that the target is
derived rather than the sole retained raw source.

## Required Clean-Rebuild Gates

The canonical rebuild must fail closed unless all of the following pass:

- official dataset version, source URL, license/citation, download metadata, and
  raw source SHA-256 are present;
- Sleep-EDF canonical person identity overlap is zero for train-test,
  train-validation, and validation-test;
- both nights from each Sleep Cassette person remain in one split;
- sampling rate, physical unit, channel identity, reference, masks, and complete
  record length are explicit per record;
- mixed rates or units are rejected or transformed by a declared, tested,
  provenance-preserving physical contract;
- channel intersection, padding, dropping, and remapping are explicit; silent
  channel-count truncation is rejected;
- record/window limits are declared and scientifically justified; silent
  truncation to 16384 samples is rejected;
- transforms are fit only on training data or causal context as specified;
- required chief and ladder baselines are present under the same subjects,
  anchors, masks, targets, metadata allowance, tuning budget, and scorer;
- split, overlap, causality, calibration, negative-control, checksum,
  distributed, final-gate, and independent reproduction reports pass.

## Quarantine Release Criteria

An invalid artifact is never made claim-eligible merely by renaming it. Release
from quarantine requires a documented finding that it was incorrectly associated
with an invalid run or an exact byte-for-byte source role that remains valid.
Derived artifacts affected by split leakage, truncation, mixed physical
contracts, missing hashes, or missing baselines must be rebuilt, not released.

The invalid metrics remain only in the invalidated result registry. Canonical
tables and automated result discovery must filter all records whose
`validity_status` is not `valid` or whose `claim_eligible` value is not `true`.
