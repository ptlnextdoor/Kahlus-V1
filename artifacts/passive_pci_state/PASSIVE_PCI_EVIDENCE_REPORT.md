# Passive PCI state-discrimination gate

- claim_scope: `passive_pci_sleep_state_discrimination_complexity_beyond_spectral_baseline_subject_held_out_public_sleep_edf_cassette_not_tms_pci_not_clinical`
- stop_reason: Passive PCI gate failed on real Sleep-EDF cassette; do not claim passive complexity beats spectral baseline.
- gate_passed: **False**
- sleep_edf_status: `evaluated`
- bootstrap_mode: `claim`

## Synthetic known / null

- known wake RFS bits: 0.0321
- null wake RFS bits: -0.0019

## Sleep-EDF real cohort

- windows: 413828 subjects: 78
- wake: positive_windows=284633 residual_rfs_bits=-0.3304 ci=[-0.3494, -0.3129]
- nrem: positive_windows=103394 residual_rfs_bits=-0.2840 ci=[-0.2974, -0.2707]
- rem: positive_windows=25801 residual_rfs_bits=-0.1614 ci=[-0.1765, -0.1460]
