# Kahlus Forecastability Trial 0 - M2 Evidence Report

Gate passed: `True`
Synthetic sleep machinery passed: `True`
Real Sleep-EDF status: `parsed_local_sleep_edf`

## Synthetic Sleep Transition Hazard

- rows/events/event-patients: `1960` / `294` / `14`
- RFS bits: `0.031329` CI `[ 0.024204, 0.038342 ]`
- shuffled-target RFS bits: `-0.005941`
- time-shift RFS bits: `-0.008517`

## Held-Out Channel Reconstruction

- ridge MSE: `0.006632`
- train-mean MSE: `0.096770`
- R2: `0.931461`

## Real Sleep-EDF Subset

- pairs used: `6`
- gate failures: `none`
- rows/events/event-patients: `16682` / `929` / `6`
- RFS bits: `0.007443` CI `[ 0.000350, 0.012583 ]`
- shuffled-target RFS bits: `-0.004488`
- channel reconstruction MSE: `98.608432` vs mean `102.596339`

M2 passed on a tiny real Sleep-EDF subset. This is a machinery gate, not a publication-scale sleep result.
