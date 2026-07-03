# Kahlus Forecastability Trial 0 - M1 Evidence Report

Gate passed: `True`

## Synthetic Known Signal

- rows/events/event-patients: `1440` / `391` / `12`
- logistic RFS bits: `0.219577` CI `[ 0.198156, 0.243070 ]`
- GBM RFS bits: `0.206253`
- shuffled-target RFS bits: `-0.046156`
- time-shift RFS bits: `0.011267`
- gated baseline: `logistic_nuisance` NLL `0.540294`
- baseline/moving/random/alarm NLL: `0.540294` / `0.730654` / `0.584772` / `0.746306`

## Synthetic Null

- rows/events/event-patients: `1440` / `186` / `12`
- logistic RFS bits: `0.000732` CI `[ -0.002000, 0.003475 ]`
- GBM RFS bits: `-0.026992`
- shuffled-target RFS bits: `-0.002578`
- time-shift RFS bits: `-0.002448`
- gated baseline: `logistic_nuisance` NLL `0.374160`
- baseline/moving/random/alarm NLL: `0.374160` / `0.544230` / `0.384799` / `0.404626`

M1 stops here. M2 should not start until this instrumentation gate is reviewed.
