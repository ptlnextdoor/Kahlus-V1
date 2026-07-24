# Propofol PCI state-discrimination gate

- claim_scope: `propofol_sedation_state_discrimination_complexity_beyond_spectral_baseline_subject_held_out_openneuro_ds005620_not_tms_pci_not_clinical`
- stop_reason: Propofol PCI gate failed on ds005620; do not claim passive complexity beats spectral baseline under propofol sedation.
- gate_passed: **False**
- ds005620_status: `evaluated`
- bootstrap_mode: `claim`
- epoch_seconds: 10.0

## Synthetic known / null

- known awake RFS bits: 0.0434
- null awake RFS bits: 0.0027

## ds005620 real cohort

- windows: 2527 subjects: 21
- awake: positive_windows=630 residual_rfs_bits=-0.2072 ci=[-0.3084, -0.1208]
- sedated: positive_windows=1897 residual_rfs_bits=-0.2072 ci=[-0.3077, -0.1198]
