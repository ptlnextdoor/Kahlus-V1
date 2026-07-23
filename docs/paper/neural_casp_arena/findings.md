# Neural-CASP findings (F0–F6)

Mirrors [`docs/results/findings-ledger.md`](../../results/findings-ledger.md).
Each row is publishable — positives and negatives count equally when the
courtroom machinery is validated.

## Gate predicate

```text
GatePass = split_disjoint ∧ finite ∧ beat_best_baseline ∧ controls_pass ∧ powered ∧ scoped_claim
```

## Findings table

| ID | Finding | Verdict | Substrate / power | Headline number | Artifact | Code | Tag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F0 | Neural-CASP gate suite (M0–M5) deterministic and leakage-audited | Arena works | Synthetic + public smoke | M0/M1/M5 pass; M3/M4 fail honestly | `artifacts/forecastability_trial0_m0..m5/` | `src/neurotwin/forecastability/m0.py..m5.py` | `finding/neural-casp-gate-suite-v1` |
| F1 | Forecast "skill" 3.116 MSE / 0.972 r is overlap illusion | Invalidated | MOABB EEG v1 sidecar | Collapses on strictly-future scoring | `docs/research/eval_leakage_audit_2026-07-21.md` | `src/stf/*` (`155a115f`) | `finding/forecast-overlap-illusion-v1` |
| F2 | GRU does not beat trivial baselines under isolated forecasting | Negative | Sleep-EDF + BNCI, subject-held-out | GRU loses to persistence/ridge at h=1 | `artifacts/ridge_bnci_real/` | `scripts/bnci_isolated_forecast_check.py` | `finding/isolated-forecast-negative-v1` |
| F3 | Peripheral channels add no residual forecastability for transitions | Negative (scout) | Sleep-EDF smoke; synthetic OK | Real RFS indistinguishable from null | `artifacts/interoception_rfs_scout/` | `interoception_scout.py` | `finding/interoception-rfs-scout-v1` |
| F4 | Passive complexity does not beat spectral for wake/NREM/REM | Powered negative | 78 subj, 413,828 windows, n_boot=2000 | Wake -0.330; NREM -0.284; REM -0.161 bits | `artifacts/passive_pci_state/` | `passive_pci.py`, `complexity_features.py` | `finding/passive-pci-negative-v1` |
| F5 | Propofol sedation: complexity beyond spectral for awake vs sedated | Powered negative | 12/21 subjects, 1298 windows, n_boot=2000 | Awake −0.174 [−0.314, −0.050]; sedated −0.174 [−0.306, −0.047] bits | `artifacts/propofol_pci_state/` | `propofol_pci.py` | `finding/propofol-pci-powered-negative-v1` |
| F6 | Autonomic block beyond cortical spectral for micro-arousal | Synthetic OK; NSRR loader fixed | Claim bootstrap synthetic; stub-EDF loader tested; MESA/SHHS when credentialed | Known RFS 0.021; real cohort pending NSRR download | `artifacts/autonomic_rfs_arousal/` | `autonomic_rfs.py`, `_rfs_eval.py` | `finding/autonomic-rfs-pending-nsrr-v1` |

## Joint interpretation

Every real-data residual probe through F4 is a **negative** on machinery that
passes synthetic known/null validation. Supported thesis:

> Under strict subject-held-out evaluation and the best trivial baseline ladder,
> headline residual forecastability in noninvasive EEG literature does not
> reproduce. F1 shows one mechanism (input–target overlap).

The arena (F0) is the asset. F1–F4 are evidence it bites. F5 tests whether
propofol sedation — PCI's natural substrate — changes the sign.

## Claim scopes (committed in artifacts)

- F3: `peripheral_autonomic_residual_forecastability_sleep_state_transitions_public_sleep_edf_smoke_not_gastric_egg_not_coleman_data_scout_grade`
- F4: `passive_pci_sleep_state_discrimination_complexity_beyond_spectral_baseline_subject_held_out_public_sleep_edf_cassette_not_tms_pci_not_clinical`
- F5: `propofol_sedation_state_discrimination_complexity_beyond_spectral_baseline_subject_held_out_openneuro_ds005620_not_tms_pci_not_clinical`
