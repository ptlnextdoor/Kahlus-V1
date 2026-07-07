# Kahlus Forecastability Trial 0 - M3 Evidence Report

Gate passed: `False`
Forecastability class: `UNDERPOWERED`
Gate failures: `underpowered_event_patients, chbmit_underpowered_event_patients, primary_not_better_than_gated_baseline, chbmit_primary_not_better_than_gated_baseline, primary_rfs_ci_includes_zero, chbmit_primary_rfs_ci_includes_zero, external_dataset_held_out_not_run`

## Source Audit

- CHB-MIT: `local_manifest`
- external scalp corpus: `not_run_requires_external_tusz_access`
- primary horizon: `300` seconds

## CHB-MIT Development Smoke

- recordings: `3`
- rows/events/event-patients: `7082` / `730` / `3`
- RFS bits: `-0.388076` CI `[ -0.460189, -0.325381 ]`
- gated baseline: `moving_average` NLL `0.020901`
- GBM RFS bits: `-0.572686`
- shuffled-target RFS bits: `-0.463231`
- time-shift RFS bits: `-0.376830`
- nuisance probe patient/site/time/session accuracy: `0.410419` / `1.000000` / `0.152901` / `0.410419`

## TUSZ External Held-Out

- status: `not_run_no_local_tusz_root`

## Final Verdict

- gate_passed: `False`
- forecastability_class: `UNDERPOWERED`
- claim boundary: research forecastability screening only; no clinical seizure prediction claim is permitted.

M3 stops here. No clinical seizure prediction claim is permitted from this gate.
