# Kahlus Forecastability Trial 0 - M4 Evidence Report

Gate passed: `False`
Synthetic gate passed: `True`
Sleep-EDF smoke status: `completed_sleep_edf_smoke`
Sleep-EDF smoke failures: `sleep_edf_underpowered_event_patients, sleep_edf_time_shift_control_too_close`

## Method

Leakage-safe forecastability-vs-horizon curve: labels are shifted within each patient only, then RFS is recomputed per horizon against the strongest gated nuisance/trivial baseline.

## Synthetic Known Signal

| horizon | RFS bits | CI low | CI high | events | event patients | gated baseline | shuffle | time-shift |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | 0.219577 | 0.198156 | 0.243070 | 391 | 12 | logistic_nuisance | -0.046156 | 0.011267 |
| 2 | 0.132488 | 0.110707 | 0.157288 | 388 | 12 | logistic_nuisance | -0.033590 | 0.025028 |
| 3 | 0.081924 | 0.061605 | 0.105478 | 387 | 12 | logistic_nuisance | -0.027573 | 0.031470 |

- positive-RFS AUC: `0.144663` bits

## Synthetic Null

| horizon | RFS bits | CI low | CI high | events | event patients | gated baseline | shuffle | time-shift |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | 0.000732 | -0.002000 | 0.003475 | 186 | 12 | logistic_nuisance | -0.002578 | -0.002448 |
| 2 | -0.001044 | -0.003510 | 0.000829 | 186 | 12 | logistic_nuisance | 0.000019 | 0.000074 |
| 3 | 0.000503 | -0.002595 | 0.003367 | 185 | 12 | logistic_nuisance | -0.000464 | -0.000881 |

- positive-RFS AUC: `0.000412` bits

## Sleep-EDF Smoke

| horizon | RFS bits | CI low | CI high | events | event patients | gated baseline | shuffle | time-shift |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | 0.007443 | 0.001842 | 0.012144 | 929 | 6 | alarm_time_surrogate | -0.005251 | 0.008158 |
| 2 | 0.004221 | -0.000447 | 0.008033 | 929 | 6 | alarm_time_surrogate | -0.009122 | 0.004972 |
| 3 | 0.002333 | -0.002356 | 0.006692 | 929 | 6 | alarm_time_surrogate | -0.010490 | 0.002764 |

- positive-RFS AUC: `0.004666` bits

M4 is a benchmark-method gate only; no clinical or foundation-model claim is permitted.
