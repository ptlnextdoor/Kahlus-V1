# Invalidated Result Registry

**Registry version:** 1.1.0
**Effective date:** 2026-07-15
**Canonical protocol:** `kahlus.hnph.phase0.v0.4`

This registry records exact historical facts without turning invalid metrics into
scientific evidence. Invalidated artifacts remain available for forensic audit
under quarantine, but they cannot be selected, compared with canonical results,
or used in claims.

Records INV-001 through INV-003 are invalid results. INV-004 is a protocol
supersession with no result attached; it is recorded here because the frozen B2
addendum requires a superseding change to carry a registry entry.

## INV-001: Historical 3.116 Result

**Status:** Invalid for future forecasting and HNPH claims.

- Reported future-task MSE: `3.11607456072082` (historically shortened to
  `3.116`).
- The prepared task used a 128-sample window transformed into 127-sample input
  and target sequences sharing 126 samples.
- The instantiated path was `NeuralStateSpaceTranslator` with the GRU-backed
  `ssm_fallback`.
- It is therefore an overlapping-target GRU translator result. It is not a
  distinct future-window result and must not be attributed to NFC, Mamba, or a
  true SSM.
- It predates and is incompatible with `kahlus.hnph.phase0.v0.2`.

Permitted wording: "Historical overlapping-target GRU translator result."

## INV-002: Failed Seven-A100 Run

**Status:** Invalid failed execution; no claim-bearing metric.

- Requested world size: `7` A100 ranks.
- Failure: distributed rank drift, followed by collective/NCCL timeout.
- Completion and rank-consistency gates did not pass.
- No metric from partial or rank-local output is eligible for scientific use.

Permitted wording: "The seven-A100 run failed due to distributed rank drift."

## INV-003: Completed Six-A100 Run

**Status:** Invalid despite process completion.

- Reported test MSE: `354076727.1019103`.
- Reported test Pearson `r`: `0.6294566387`.
- Final evidence gate: `false`.
- Sleep-EDF canonical person overlap counts were `26` train-test, `18`
  train-validation, and `8` validation-test.
- Channel counts were silently truncated.
- Records with mixed sampling rates and mixed physical units were combined
  without a valid physical harmonization contract.
- Record/sample content was truncated to `16384` samples.
- Required source hashes were missing.
- Required baseline outputs were missing.

Completion does not cure leakage, physical-data corruption, missing provenance,
or missing comparators. The metrics above are retained only as invalidation
facts and may not appear in a canonical result table.

Permitted wording: "A completed six-A100 run was invalidated by split leakage,
silent truncation, mixed physical contracts, and missing evidence; its final
gate was false."

## INV-004: HNPH v0.3 Protocol Superseded by v0.4

**Status:** Superseded for future claim-mode work; not invalid, and not
executed.

- Superseded protocol: `kahlus.hnph.phase0.v0.3`.
- Superseding protocol: `kahlus.hnph.phase0.v0.4`, frozen at SHA-256
  `401a8e47db3aefc5c549fec956254f291f71e43c6bf636c464551df552acc839`.
- Claim-mode runs executed under v0.3: `0`. No metric exists to invalidate, and
  nothing under v0.3 is quarantined.
- v0.3 and its B2 preregistration addendum are preserved unchanged for audit.
  v0.3 remains the correct historical record of what was frozen at the time.
- v0.4 supersedes v0.3 by requiring hash-bound DOD source qualification: five
  independent rater streams per dataset, person identities, physical units,
  sampling rates, licenses, and source/annotation hashes, verified locally and
  fail-closed before any migration.
- Source qualification status: `unverified`. DOD-O remains sealed
  (`external_test_opened: false`). Migration, training, and claim-mode
  evaluation are therefore unauthorized.

This entry exists because `docs/research/hnph_b2_preregistration_addendum.md`
freezes v0.3 with the condition that "a later change requires a new addendum
version and an invalidation-registry entry". The addendum half is
`docs/research/hnph_v0.4_source_qualification_addendum.md`; this is the registry
half. Without it the v0.4 migration would violate the freeze rule the project
set for itself.

Permitted wording: "HNPH protocol v0.3 is preserved for audit and superseded by
v0.4 for all future claim-mode work; no claim-mode result was computed under
v0.3."

## Registry-Wide Disposition

- Preserve raw source caches unchanged and read-only where feasible.
- Quarantine all derived prepared data, checkpoints, predictions, metrics,
  figures, and evidence bundles associated with INV-001 through INV-003.
- INV-004 carries no quarantine: it records a protocol supersession, not a
  result. Nothing was computed under v0.3.
- Do not delete quarantined material until hashes and provenance are inventoried.
- Do not warm-start, tune, select, compare, or report from quarantined material.
- A clean rebuild from verified raw sources is required; changing a label or
  regenerating only the final report cannot rehabilitate these results.
