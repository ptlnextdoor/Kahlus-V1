# Kahlus Forecastability Trial 0 - M2 Evidence Report

Gate passed: `True`
Synthetic sleep machinery passed: `True`
Real Sleep-EDF status: `parsed_local_sleep_edf`

## Synthetic Sleep Transition Hazard

- rows/events/event-patients: `1960` / `296` / `14`
- RFS bits: `0.022743` CI `[ 0.013903, 0.032134 ]`
- shuffled-target RFS bits: `-0.005072`
- time-shift RFS bits: `0.002358`

## Held-Out Channel Reconstruction

- ridge MSE: `0.006442`
- train-mean MSE: `0.073033`
- R2: `0.911781`

## Real Sleep-EDF Subset

- pairs used: `6`
- gate failures: `none`
- rows/events/event-patients: `16682` / `929` / `6`
- RFS bits: `0.000374` CI `[ 0.000164, 0.000599 ]`
- shuffled-target RFS bits: `-0.004489`
- channel reconstruction MSE: `98.608432` vs mean `102.596339`

M2 passed on a tiny real Sleep-EDF subset. This is a machinery gate, not a publication-scale sleep result.
