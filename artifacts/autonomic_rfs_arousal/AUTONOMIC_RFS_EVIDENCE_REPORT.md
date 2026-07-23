# Autonomic RFS arousal gate

- claim_scope: `autonomic_residual_forecastability_micro_arousal_beyond_cortical_spectral_subject_held_out_nsrr_mesa_shhs_not_gastric_egg_not_coleman_data_not_clinical`
- stop_reason: Synthetic fixture validated; MESA real cohort not available — obtain NSRR credentialed access before claim-grade autonomic RFS.
- gate_passed: **True**
- mesa_status: `skipped`
- shhs_status: `skipped`
- bootstrap_mode: `claim`
- epoch_seconds: 30.0

## Estimand

- Y: micro_arousal_within_horizon_epochs
- B: cortical_spectral_plus_cycle_history
- Z: autonomic_hrv_resp_eog_emg_infraslow

## Synthetic known / null

- known primary RFS bits: 0.0207
- null primary RFS bits: -0.0001

## MESA real cohort

- not run: ['mesa_root_missing']

## SHHS dataset-held-out

- not run: ['shhs_root_missing']
