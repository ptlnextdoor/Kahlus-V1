# Neural-CASP benchmark contribution

Neural-CASP is the **named, citable benchmark artifact** — not any single model.

## What reviewers cite

| Component | Location | Role |
| --- | --- | --- |
| RFS bits estimator | `m1._crossfit_residual_proba`, `_rfs_payload` | Cross-fitted I(Y;Z\|B) in bits |
| Baseline ladder | `m1._best_baseline`, `_moving_average_proba` | Must beat best trivial |
| Cluster bootstrap CI | `m1._cluster_bootstrap_rfs` | Claim mode n_boot≥2000 |
| Cluster permutation | `m4._cluster_permutation_rfs` | Sign-flip null on patient clusters |
| Control contract | `passive_pci.py`, `autonomic_rfs.py` | Shuffle, time-shift, circular-shift, subject-probe |
| Overlap/copy-trap audit | F1 + `amrith_acceptance.py` + M0 gate | 127→128 constitutionalized |
| Evidence pattern | All `run_*_gate.py` scripts | `runner → gate → JSON + Markdown` |

## Defendant dock

Each hypothesis plugs into the same courtroom:

- F2 isolated forecast (GRU)
- F3 interoception scout (EOG/EMG/resp)
- F4 Passive PCI (complexity vs spectral, sleep stages)
- F5 propofol PCI (awake vs sedated)
- F6 autonomic RFS (HRV/resp/EOG/EMG vs arousal)

Honest negatives are first-class outputs. A benchmark that only flatters its own model is worthless.

## Submission target

NeurIPS **Datasets & Benchmarks** — see [`../neurips_2026/submission.md`](../neurips_2026/submission.md).
