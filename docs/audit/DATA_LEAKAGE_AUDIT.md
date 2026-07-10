# Data Leakage and Pseudoreplication Audit

## Executive Finding

The recovered split appears subject-disjoint, but the central task has direct context-target sample overlap. This is task leakage in the scientific sense even though train and test subjects are separated.

## Leakage Classes

| Risk | Current control | Residual defect | Severity |
|---|---|---|---|
| Subject/patient overlap | manifest audits compare IDs | raw identity is not independently hashed; duplicate public identities need explicit maps | P1 |
| Recording overlap | record IDs checked across splits | duplicate IDs can overwrite before validation; source-file identity incomplete | P1 |
| Window overlap across splits | range audits exist | no mandatory temporal embargo for time-based splits | P1 |
| Context-target overlap | none sufficient | 126/127 target samples are present in context | P0 |
| Temporal-neighbor leakage | some stride diagnostics | stride is mistaken for context-target separation in tests | P0 |
| Normalization leakage | contract hooks exist | exact fitted parameters and train-only lineage not always persisted | P1 |
| Feature/imputation leakage | gates inspect some metadata | no complete transform DAG from train fit to test apply | P1 |
| Hyperparameter leakage | configs preserve settings | no nested model-selection protocol for main result | P0/P1 |
| Pretraining contamination | not relevant to recovered GRU | required before LaBraM/BIOT use | P2 |
| Dataset duplicate recordings | dataset registry work exists | no cross-dataset content-hash deduplication | P1 |

## Exact Target Defect

In `prepared_tasks._future_xy`, a sequence `s[0:128]` produces:

- input `s[0:127]`
- target `s[1:128]`

The target values `s[1:127]` are already present in the input. A model can obtain low MSE by shifting or copying the observed waveform. In `_future_windows`, `forecast_horizon=1` has the same structure for arbitrary window length.

Required invariant:

`target_start >= input_start + input_length + embargo`.

This invariant must be checked in task construction, manifests, gates, and tests.

## Pseudoreplication

EEG elements and windows are not independent experimental units. Primary uncertainty must resample subjects or patients; secondary hierarchical resampling can nest recordings and sessions within subjects. Elementwise bootstrap results may be retained only as descriptive numerical uncertainty and must not support scientific significance.

## Required Corruption Tests

- Duplicate one raw file under a new filename and require content-hash rejection.
- Assign one subject to train and test and require gate failure.
- Place adjacent windows across splits and require embargo failure.
- Reuse normalization statistics computed on all subjects and require lineage failure.
- Inject subject ID, session ID, labels, or future-derived metadata and require nuisance/forbidden-field failure.
- Shuffle targets within training only and verify validation/test targets remain untouched.
