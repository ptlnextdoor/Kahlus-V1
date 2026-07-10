# Invalidated Result Registry

**Registry version:** 1
**Effective date:** 2026-07-10

This registry records exact historical facts without turning invalid metrics into
scientific evidence. Invalidated artifacts remain available for forensic audit
under quarantine, but they cannot be selected, compared with canonical results,
or used in claims.

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

## Registry-Wide Disposition

- Preserve raw source caches unchanged and read-only where feasible.
- Quarantine all derived prepared data, checkpoints, predictions, metrics,
  figures, and evidence bundles associated with INV-001 through INV-003.
- Do not delete quarantined material until hashes and provenance are inventoried.
- Do not warm-start, tune, select, compare, or report from quarantined material.
- A clean rebuild from verified raw sources is required; changing a label or
  regenerating only the final report cannot rehabilitate these results.
