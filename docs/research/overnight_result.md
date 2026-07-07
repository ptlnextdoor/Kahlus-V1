# Overnight Result - 2026-07-03

## TL;DR

Kahlus now has a stricter RFS gate and a new M4 forecastability-vs-horizon benchmark lane. The CHB-MIT smoke no longer shows positive forecastability once the gate uses the strongest trivial baseline: moving average wins, and the primary RFS is `-0.388076` bits. The M4 curve passes on synthetic known/null fixtures. The current committed M4 artifact was regenerated without a local Sleep-EDF root, so Sleep-EDF remains not run in the artifact rather than being promoted from stale local evidence.

## Precise Claim

Allowed claim: Kahlus implements an executable, leakage-aware residual forecastability instrument that can estimate a horizon-wise RFS curve under held-out patient splits and fail when nuisance/history baselines explain the signal.

Forbidden claim: no clinical seizure prediction, no diagnostic utility, no foundation-model claim, no "first" claim.

## Math

For horizon `h`, M4 estimates:

```text
RFS_h = [NLL(Y_{t+h}, q_baseline) - NLL(Y_{t+h}, q_full)] / log(2)
```

With oracle nuisance and full predictors this is the conditional information gain:

```text
I(Y_{t+h}; Z_t | B_t)
```

The implementation uses patient-safe horizon labels: for each patient group, labels are shifted only inside that group. No last row from one patient can become the future label for the next patient, and terminal rows without a within-patient future label are excluded before fitting/scoring for that horizon.

## Evidence

Artifacts:

- `artifacts/forecastability_trial0_m3/m3_gate_report.json`
- `artifacts/forecastability_trial0_m3/M3_EVIDENCE_REPORT.md`
- `artifacts/forecastability_trial0_m4/m4_gate_report.json`
- `artifacts/forecastability_trial0_m4/M4_EVIDENCE_REPORT.md`

M3 after hardened gated baseline:

- gate: `False`
- gated baseline: `moving_average`, NLL `0.020901`
- primary model NLL: `0.289895`
- primary RFS: `-0.388076` bits, CI `[ -0.449901, -0.325381 ]`
- failures include `primary_not_better_than_gated_baseline`, `primary_rfs_ci_includes_zero`, `underpowered_event_patients`, and `external_dataset_held_out_not_run`

M4 synthetic known signal:

- horizon 1 RFS: `0.219577` bits, CI `[0.198156, 0.243070]`
- horizon 2 RFS: `0.132555` bits, CI `[0.112087, 0.156228]`
- horizon 3 RFS: `0.079894` bits, CI `[0.060602, 0.102731]`
- exact patient-cluster sign-flip p-values: `0.000488`, `0.000488`, `0.000488`
- positive-RFS AUC: `0.144009` bits
- valid/invalid rows by horizon: `1440/0`, `1416/24`, `1392/48`

M4 synthetic null:

- horizon RFS values: `0.000732`, `-0.000631`, `0.000403`
- exact patient-cluster sign-flip p-values: `0.306322`, `0.715402`, `0.398584`
- all CIs straddle zero
- positive-RFS AUC: `0.000378` bits
- valid/invalid rows by horizon: `1440/0`, `1416/24`, `1392/48`

M4 Sleep-EDF smoke:

- status: `not_run_no_local_sleep_edf_root`
- gate: `False`
- failures: `sleep_edf_smoke_not_completed`
- current artifact does not include Sleep-EDF curve rows because the local subset root was unavailable during artifact refresh

## Implementation Notes

- Replaced the clipped residual GD path with a small convex offset-IRLS solver in `src/neurotwin/forecastability/m1.py`.
- RFS payloads now use the best nuisance/trivial baseline among logistic nuisance, moving average, random warning, and alarm-time surrogate.
- Added `src/neurotwin/forecastability/m4.py`.
- Added `tests/forecastability/test_m4.py`.
- M4 now reports nuisance probes per horizon and excludes terminal rows without a within-patient/session future label before fitting/scoring.
- M4 reports fixed-prediction patient-cluster sign-flip permutation p-values per horizon; only the explicit primary horizon is inferential for the gate.
- Added an M4 Sleep-EDF primary-horizon pre-analysis plan and runner contract so a future public-data run can execute from an out-of-repo raw-data root without committing local paths or raw data.
- Sleep-EDF filenames are parsed into dataset-scoped subject IDs and night/session IDs, so split/cluster units are subjects while horizon labels stop at session boundaries.
- Added an M3 artifact freshness test that recomputes failures from current gate logic.
- Added `.github/workflows/ci.yml`.

## Limitations

- M4 is a benchmark-method contribution, not a powered biological result.
- Sleep-EDF smoke is not present in the current committed M4 artifact because no local Sleep-EDF root was available at refresh time.
- The M4 Sleep-EDF runner is now preregistered as a future execution contract, but no full Sleep-EDF result is claimed by this note.
- Historical tiny Sleep-EDF M2 evidence must be treated cautiously because subject/night metadata now makes paired nights stricter than pair-count-as-subject accounting.
- CHB-MIT remains a development smoke path and fails after the stronger baseline.
- TUSZ external validation was not run because no local out-of-repo TUSZ root is available.
- Bootstrap remains percentile-based and small in the current shared helper; BCa and larger `n_boot` remain future hardening.
- The M4 permutation control is synthetic method evidence only. It does not compensate for the missing full Sleep-EDF/public-data run.

## Next Experiments

1. Run the M4 Sleep-EDF primary-horizon runner against an out-of-repo Sleep-EDF root and preserve redacted subject/night execution metadata.
2. Run CHB-MIT with at least 8 event-patients, then real out-of-repo TUSZ.
3. Add a multiplicity-controlled protocol if future M4 claims use more than the preregistered primary horizon.
4. Add variance-normalized skill for A100 aggregate reporting and stop headlining unweighted task means.
