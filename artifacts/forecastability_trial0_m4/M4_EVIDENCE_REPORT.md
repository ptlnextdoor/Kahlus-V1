# Kahlus Forecastability Trial 0 - M4 Evidence Report

Gate passed: `False`
Synthetic gate passed: `True`
Sleep-EDF smoke status: `not_run_no_local_sleep_edf_root`
Sleep-EDF smoke failures: `sleep_edf_smoke_not_completed`

## Method

Leakage-safe horizon-wise label curve: labels are shifted within each patient only, terminal rows without a within-patient future label are excluded before fitting, then RFS is recomputed per horizon against the strongest gated nuisance/trivial baseline. This is not a censoring-aware survival model.

Nuisance probes are reported for every M4 horizon and are claim blockers if accuracy exceeds chance + 0.20; passing probes do not unlock any clinical, causal, or model-superiority claim.

## Synthetic Known Signal

| horizon | total rows | valid rows | evaluated rows | invalid terminal | RFS bits | CI low | CI high | events | event patients | gated baseline | shuffle | time-shift | nuisance probes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| 1 | 1440 | 1440 | 1440 | 0 | 0.219577 | 0.198156 | 0.243070 | 391 | 12 | logistic_nuisance | -0.046156 | 0.011267 | passed (patient=0.117/0.083, site=0.537/0.500, time_bucket=0.340/0.250, session=0.533/0.500) |
| 2 | 1440 | 1428 | 1428 | 12 | 0.133120 | 0.113036 | 0.156743 | 388 | 12 | logistic_nuisance | -0.038720 | 0.028435 | passed (patient=0.101/0.083, site=0.525/0.500, time_bucket=0.334/0.250, session=0.525/0.500) |
| 3 | 1440 | 1416 | 1416 | 24 | 0.082506 | 0.063052 | 0.104817 | 387 | 12 | logistic_nuisance | -0.019520 | 0.033137 | passed (patient=0.117/0.083, site=0.534/0.500, time_bucket=0.343/0.250, session=0.553/0.500) |

- positive-RFS AUC: `0.145068` bits

## Synthetic Null

| horizon | total rows | valid rows | evaluated rows | invalid terminal | RFS bits | CI low | CI high | events | event patients | gated baseline | shuffle | time-shift | nuisance probes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| 1 | 1440 | 1440 | 1440 | 0 | 0.000732 | -0.002000 | 0.003475 | 186 | 12 | logistic_nuisance | -0.002578 | -0.002448 | passed (patient=0.087/0.083, site=0.537/0.500, time_bucket=0.319/0.250, session=0.481/0.500) |
| 2 | 1440 | 1428 | 1428 | 12 | -0.001066 | -0.003515 | 0.000791 | 186 | 12 | logistic_nuisance | -0.000611 | 0.000200 | passed (patient=0.071/0.083, site=0.517/0.500, time_bucket=0.382/0.250, session=0.489/0.500) |
| 3 | 1440 | 1416 | 1416 | 24 | 0.000518 | -0.002686 | 0.003458 | 185 | 12 | logistic_nuisance | -0.004206 | -0.000457 | passed (patient=0.104/0.083, site=0.504/0.500, time_bucket=0.356/0.250, session=0.492/0.500) |

- positive-RFS AUC: `0.000417` bits

## Sleep-EDF Smoke

- status: `not_run_no_local_sleep_edf_root`

M4 is a benchmark-method gate only; no clinical or foundation-model claim is permitted.
