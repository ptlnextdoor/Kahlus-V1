# Kahlus Forecastability Trial 0 - M1 Evidence Report

Gate passed: `True`

## Synthetic Known Signal

- rows/events/event-patients: `1440` / `383` / `12`
- logistic RFS bits: `0.235298` CI `[ 0.199965, 0.277949 ]`
- GBM RFS bits: `0.245883`
- shuffled-target RFS bits: `-0.116459`
- time-shift RFS bits: `-0.066434`
- gated baseline: `logistic_nuisance` NLL `0.520515`
- baseline/moving/random/alarm NLL: `0.520515` / `0.750297` / `0.579211` / `0.807673`

## Synthetic Null

- rows/events/event-patients: `1440` / `201` / `12`
- logistic RFS bits: `-0.000572` CI `[ -0.002406, 0.001382 ]`
- GBM RFS bits: `-0.045462`
- shuffled-target RFS bits: `-0.031868`
- time-shift RFS bits: `-0.016422`
- gated baseline: `logistic_nuisance` NLL `0.390731`
- baseline/moving/random/alarm NLL: `0.390731` / `0.584275` / `0.404206` / `0.420171`

M1 stops here. M2 should not start until this instrumentation gate is reviewed.
