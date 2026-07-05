# Overnight Result - 2026-07-03

## TL;DR

Kahlus now has a stricter RFS gate and a new M4 forecastability-vs-horizon benchmark lane. The CHB-MIT smoke no longer shows positive forecastability once the gate uses the strongest trivial baseline: moving average wins, and the primary RFS is `-0.388076` bits. The new M4 curve passes on synthetic known/null fixtures and runs on the local Sleep-EDF smoke path, but the real smoke gate fails honestly because it has only 6 event-patients and the time-shift control is too close.

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

The implementation uses patient-safe horizon labels: for each patient group, labels are shifted only inside that group. No last row from one patient can become the future label for the next patient.

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
- horizon 2 RFS: `0.132488` bits, CI `[0.110707, 0.157288]`
- horizon 3 RFS: `0.081924` bits, CI `[0.061605, 0.105478]`
- positive-RFS AUC: `0.144663` bits

M4 synthetic null:

- horizon RFS values: `0.000732`, `-0.001044`, `0.000503`
- all CIs straddle zero
- positive-RFS AUC: `0.000412` bits

M4 Sleep-EDF smoke:

- status: `completed_sleep_edf_smoke`
- gate: `False`
- failures: `sleep_edf_underpowered_event_patients`, `sleep_edf_time_shift_control_too_close`
- horizon 1 RFS: `0.007443` bits, but time-shift RFS is `0.008158`

## Implementation Notes

- Replaced the clipped residual GD path with a small convex offset-IRLS solver in `src/neurotwin/forecastability/m1.py`.
- RFS payloads now use the best nuisance/trivial baseline among logistic nuisance, moving average, random warning, and alarm-time surrogate.
- Added `src/neurotwin/forecastability/m4.py`.
- Added `tests/forecastability/test_m4.py`.
- Added an M3 artifact freshness test that recomputes failures from current gate logic.
- Added `.github/workflows/ci.yml`.

## Limitations

- M4 is a benchmark-method contribution, not a powered biological result.
- Sleep-EDF smoke has only 6 event-patients here.
- CHB-MIT remains a development smoke path and fails after the stronger baseline.
- TUSZ external validation was not run because no local out-of-repo TUSZ root is available.
- Bootstrap remains percentile-based and small in the current shared helper; BCa and larger `n_boot` remain future hardening.

## Next Experiments

1. Run full Sleep-EDF with enough event-patients and pre-register one primary horizon.
2. Run CHB-MIT with at least 8 event-patients, then real out-of-repo TUSZ.
3. Add cluster-permutation p-values for the M4 primary horizon.
4. Add nuisance probes to M4 per horizon and fail if patient/site/session are recoverable above threshold.
5. Add variance-normalized skill for A100 aggregate reporting and stop headlining unweighted task means.
