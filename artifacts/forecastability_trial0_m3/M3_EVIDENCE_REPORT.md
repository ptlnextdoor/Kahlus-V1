# Kahlus Forecastability Trial 0 - M3 Evidence Report

Gate passed: `False`
Forecastability class: `UNDERPOWERED`
Gate failures: `underpowered_event_patients, external_dataset_held_out_not_run`

## Source Audit

- CHB-MIT: `local_manifest`
- external scalp corpus: `not_run_requires_external_tusz_access`
- primary horizon: `300` seconds

## CHB-MIT Development Smoke

- recordings: `3`
- rows/events/event-patients: `7082` / `730` / `3`
- RFS bits: `0.009148` CI `[ 0.006352, 0.036343 ]`
- GBM RFS bits: `0.301703`
- shuffled-target RFS bits: `0.002462`
- time-shift RFS bits: `0.007205`

M3 stops here. No clinical seizure prediction claim is permitted from this gate.
