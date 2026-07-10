# NeurIPS Readiness Review

## Score: 2 / 10

The repository demonstrates unusual care in experiment contracts, negative controls, claim gates, packaging, and documentation. The current scientific paper is nevertheless not submission-ready because the headline task is mischaracterized and the headline architecture is not the model that produced the recovered result.

## Precise Contribution Available Today

A research-engineering framework for subject-aware neural-data manifests, falsification-oriented baseline ladders, and evidence-bundle claim gates, demonstrated mainly on synthetic fixtures and one recovered overlapping-target EEG sequence task.

## Novelty Assessment

- Architectural novelty: proposed but unvalidated.
- Theoretical novelty: none established.
- Empirical novelty: not established under a valid distinct-future task.
- Infrastructural novelty: potentially useful, but must be compared to MOABB, MNE/Braindecode, and benchmark/evaluation systems.
- Translational novelty: unsupported.

## Likely Rejection Reasons

1. The future target is substantially present in the input.
2. The reported result is attributed to a different named architecture.
3. Baselines are not tuned or compute-matched.
4. Statistics use the wrong independent unit.
5. No external dataset/site validation exists.
6. Neural-field/operator terminology is not supported by coordinates or discretization experiments.
7. The strongest real masked-reconstruction result fails.
8. The evidence bundle is not raw-to-result reproducible.

## Narrowest Publishable Paper

After correction, the strongest candidate is not a clinical or foundation-model paper. It is a leakage-aware EEG forecasting benchmark paper asking when nonlinear models add predictive information beyond autocorrelation under strictly non-overlapping, subject- and dataset-held-out evaluation. ResearchDock can be presented as the reproducibility/falsification mechanism if its value is empirically demonstrated by catching invalid protocols.

## Highest-Information Experiment

Re-run BNCI2014-001 with strictly non-overlapping targets at several gaps/horizons, subject-level paired intervals, and matched persistence/ridge/AR/TCN/GRU/Transformer controls. If the advantage survives, repeat the frozen protocol on EEGMMI or another independently sourced dataset. If it disappears, the current central model claim is falsified, which is itself an important correction.

## Submission Threshold

P0 issues must be fixed, a preregistered/frozen task contract must pass, at least two datasets must complete with subject-level inference, and the architecture name must match the executable model and ablations before a serious NeurIPS submission.
