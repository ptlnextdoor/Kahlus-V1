# Propofol PCI state-discrimination gate

- claim_scope: `propofol_sedation_state_discrimination_complexity_beyond_spectral_baseline_subject_held_out_openneuro_ds005620_not_tms_pci_not_clinical`
- stop_reason: ds005620 cohort underpowered (2 subjects < 8); do not claim powered propofol PCI result.
- gate_passed: **False**
- ds005620_status: `evaluated`
- bootstrap_mode: `claim`
- epoch_seconds: 10.0

## Synthetic known / null

- known awake RFS bits: 0.0434
- null awake RFS bits: 0.0027

## ds005620 real cohort

- windows: 178 subjects: 2
- awake: positive_windows=33 residual_rfs_bits=-0.2062 ci=[-0.4128, 0.5067]
- sedated: positive_windows=145 residual_rfs_bits=-0.2062 ci=[-0.4128, 0.5067]
