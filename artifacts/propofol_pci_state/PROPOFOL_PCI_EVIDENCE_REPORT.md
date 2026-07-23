# Propofol PCI state-discrimination gate

- claim_scope: `propofol_sedation_state_discrimination_complexity_beyond_spectral_baseline_subject_held_out_openneuro_ds005620_not_tms_pci_not_clinical`
- stop_reason: ds005620 cohort underpowered (7 subjects < 8); do not claim powered propofol PCI result.
- gate_passed: **False**
- ds005620_status: `evaluated`
- bootstrap_mode: `claim`
- epoch_seconds: 10.0

## Synthetic known / null

- known awake RFS bits: 0.0434
- null awake RFS bits: 0.0027

## ds005620 real cohort

- windows: 854 subjects: 7
- awake: positive_windows=210 residual_rfs_bits=-0.2682 ci=[-0.4349, -0.1211]
- sedated: positive_windows=644 residual_rfs_bits=-0.2682 ci=[-0.4329, -0.1216]
