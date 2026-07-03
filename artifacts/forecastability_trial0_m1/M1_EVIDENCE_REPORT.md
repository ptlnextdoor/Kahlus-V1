# Kahlus Forecastability Trial 0 - M1 Evidence Report

Gate passed: `True`

## Synthetic Known Signal

- rows/events/event-patients: `1440` / `383` / `12`
- logistic RFS bits: `0.219874` CI `[ 0.181921, 0.269215 ]`
- GBM RFS bits: `0.236082`
- shuffled-target RFS bits: `-0.109291`
- time-shift RFS bits: `-0.057310`
- baseline/moving/random/alarm NLL: `0.513722` / `0.750297` / `0.579211` / `0.852386`

## Synthetic Null

- rows/events/event-patients: `1440` / `201` / `12`
- logistic RFS bits: `-0.000841` CI `[ -0.001737, 0.000084 ]`
- GBM RFS bits: `-0.045538`
- shuffled-target RFS bits: `-0.000958`
- time-shift RFS bits: `0.000847`
- baseline/moving/random/alarm NLL: `0.390679` / `0.584275` / `0.404206` / `0.437333`

M1 stops here. M2 should not start until this instrumentation gate is reviewed.
