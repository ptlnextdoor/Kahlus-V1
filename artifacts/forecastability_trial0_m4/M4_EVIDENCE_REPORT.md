# Kahlus Forecastability Trial 0 - M4 Evidence Report

Gate passed: `False`
Synthetic gate passed: `True`
Sleep-EDF smoke status: `not_run_no_local_sleep_edf_root`
Sleep-EDF smoke failures: `sleep_edf_smoke_not_completed`

## Method

Leakage-aware horizon-wise label curve: labels are shifted within each patient/session only, terminal rows without a within-patient/session future label are excluded before fitting, then RFS is recomputed per horizon against the strongest gated nuisance/trivial baseline. This is not a censoring-aware survival model.

Nuisance probes are reported for every M4 horizon and are claim blockers if accuracy exceeds chance + 0.20; passing probes do not unlock any clinical, causal, or model-superiority claim.

M4 also reports patient-cluster sign-flip permutation p-values for horizon RFS. Only the preregistered primary horizon is inferential for the gate; other horizons remain descriptive unless a multiplicity-controlled protocol is added.

## Synthetic Known Signal

| horizon | total rows | valid rows | evaluated rows | invalid terminal | RFS bits | CI low | CI high | events | event patients | gated baseline | shuffle | time-shift | cluster p | nuisance probes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| 1 | 1440 | 1440 | 1440 | 0 | 0.219577 | 0.198156 | 0.243070 | 391 | 12 | logistic_nuisance | -0.046156 | 0.011267 | 0.000488 | passed (patient=0.117/0.083, site=0.537/0.500, time_bucket=0.340/0.250, session=0.533/0.500) |
| 2 | 1440 | 1416 | 1416 | 24 | 0.132555 | 0.112087 | 0.156228 | 384 | 12 | logistic_nuisance | -0.031277 | 0.033745 | 0.000488 | passed (patient=0.125/0.083, site=0.519/0.500, time_bucket=0.356/0.250, session=0.553/0.500) |
| 3 | 1440 | 1392 | 1392 | 48 | 0.079894 | 0.060602 | 0.102731 | 378 | 12 | logistic_nuisance | -0.018452 | 0.035946 | 0.000488 | passed (patient=0.103/0.083, site=0.532/0.500, time_bucket=0.349/0.250, session=0.554/0.500) |

- positive-RFS AUC: `0.144009` bits

## Synthetic Null

| horizon | total rows | valid rows | evaluated rows | invalid terminal | RFS bits | CI low | CI high | events | event patients | gated baseline | shuffle | time-shift | cluster p | nuisance probes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| 1 | 1440 | 1440 | 1440 | 0 | 0.000732 | -0.002000 | 0.003475 | 186 | 12 | logistic_nuisance | -0.002578 | -0.002448 | 0.306322 | passed (patient=0.087/0.083, site=0.537/0.500, time_bucket=0.319/0.250, session=0.481/0.500) |
| 2 | 1440 | 1416 | 1416 | 24 | -0.000631 | -0.002681 | 0.001173 | 184 | 12 | logistic_nuisance | -0.000534 | 0.000337 | 0.715402 | passed (patient=0.087/0.083, site=0.540/0.500, time_bucket=0.354/0.250, session=0.479/0.500) |
| 3 | 1440 | 1392 | 1392 | 48 | 0.000403 | -0.002782 | 0.003156 | 182 | 12 | logistic_nuisance | 0.000116 | -0.000359 | 0.398584 | passed (patient=0.093/0.083, site=0.556/0.500, time_bucket=0.349/0.250, session=0.478/0.500) |

- positive-RFS AUC: `0.000378` bits

## Sleep-EDF Smoke

- status: `not_run_no_local_sleep_edf_root`

M4 is a benchmark-method gate only; no clinical or foundation-model claim is permitted.
