# Kahlus/NeuroTwin System Inventory

Audit date: 2026-07-10

Audited commit: `8fc98f32e0fe0fe4d004da6ba226fdf8316ea627`

Audit branch: `audit/forensic-prepublication`

## Frozen State

- The audited worktree was clean before this report was created.
- The only open pull request was PR 32, `codex/eeg-leakage-manuscript-docs`, based on `main`. It was not merged.
- No raw neural data or model checkpoints are committed.
- The recovered A100 evidence release was inspected read-only. No cluster job was launched.
- The local shell reported Python 3.13.11. The test environment used `/opt/miniconda3/bin/python` and requires `PYTHONPATH=src` to avoid a stale editable installation from another Kahlus worktree.
- Core dependency bounds are broad except for `scikit-learn==1.7.2`; there is no complete lockfile.

## Scale

The repository contains approximately 552 relevant files: 189 source files, 118 tests, 86 scripts, 39 configs, 90 documentation files, and 27 tracked evidence/artifact files. `src/`, `tests/`, and `scripts/` contain about 68,000 lines.

## Subsystems

| Subsystem | Scientific purpose | Principal inputs | Outputs | Tests/evidence | Audit classification |
|---|---|---|---|---|---|
| Data schemas and manifests | Standardize neural observations and provenance | arrays, subject/session/record metadata | `NeuralEventBatch`, recording and split manifests | extensive unit tests; partial real-data manifests | IMPLEMENTED_BUT_NOT_VALIDATED |
| MOABB adapter | Import public BCI trials | MOABB/MNE datasets | internal trials/manifests | adapter tests; BNCI2014-001 recovered run | IMPLEMENTED_BUT_NOT_VALIDATED |
| Public EDF adapters | Import Sleep-EDF, CHB-MIT, EEGMMI, Siena | EDF and annotations | prepared manifests/windows | synthetic/tiny loader tests; cluster preparation report | IMPLEMENTED_BUT_NOT_VALIDATED |
| Split and leakage audit | Separate subjects/records and detect reuse | recording manifests | split manifests and audit reports | strong unit coverage; incomplete temporal embargo/provenance | PARTIALLY_IMPLEMENTED |
| EEG v1 task builder | Construct forecasting pairs | continuous recordings/windows | input-target tensors | many tests; central target-overlap defect | CONTRADICTED_BY_EVIDENCE |
| Prepared-manifest task builder | Construct benchmark tasks from prepared windows | prepared window files | supervised tasks | tests; same one-sample-shift target issue | CONTRADICTED_BY_EVIDENCE |
| Classical baselines | Persistence, ridge, AR/VAR-like comparisons | supervised tasks | predictions and metrics | synthetic and recovered real evidence | IMPLEMENTED_BUT_NOT_VALIDATED |
| Neural baselines | Tiny GRU/TCN/Transformer families | supervised tasks | predictions and metrics | runnable tests; budgets not competitive | IMPLEMENTED_BUT_NOT_VALIDATED |
| `NeuralStateSpaceTranslator` | Encode, evolve, and render sequences | modality tensors | target sequence | recovered 100k-step EEG run | IMPLEMENTED_BUT_NOT_VALIDATED |
| Experimental NFC | Latent tensor dynamics and observation heads | one selected modality plus optional priors | modality prediction, latent tensor, uncertainty score | synthetic/unit evidence only | SYNTHETIC_ONLY |
| Forecastability M0-M5 | Falsification-oriented benchmark ladder | synthetic and tiny public datasets | gates and reports | M0/M1/M4/M5 synthetic; M2 tiny; M3 underpowered negative | INFRASTRUCTURE_ONLY |
| STF / dual-field / transition gym | Explore stimulus, temporal, and counterfactual structure | mostly synthetic fixtures | diagnostics and benchmark reports | synthetic evidence | SYNTHETIC_ONLY |
| Neurovisual | Symptom-map and neurovisual contracts | metadata/fixtures | schemas and reports | local evidence coverage, no clinical cohort | INFRASTRUCTURE_ONLY |
| EM branch | Electromagnetic metadata/simulation contracts | synthetic metadata | simulation/evidence records | synthetic tests | INFRASTRUCTURE_ONLY |
| ResearchDock | Experiment contracts, gates, provenance, bundles | configs and result artifacts | evidence bundles and claim gates | strong software tests; no independent reproduction | INFRASTRUCTURE_ONLY |
| A100 handoff | Package distributed experiments | source/config/data manifest | runner bundles and evidence | 3-GPU finalization artifact; failed 7-GPU training history | IMPLEMENTED_BUT_NOT_VALIDATED |

## God Nodes and Concentration Risk

The Graphify report identifies unusually connected modules around `EEGV1SprintATests`, `NumpyRidgeBaseline`, data schemas/splits, baseline implementations, and distributed training. The largest files include a 6,165-line EEG v1 test module and a 2,725-line reporting module. These concentrations increase the chance that task definitions, gates, and tests reproduce the same mistaken assumption.

## Duplicate, Abandoned, or Misleading Paths

- `tiny_ssm`, `ssm_fallback`, and `ssm` resolve to a GRU rather than an SSM implementation.
- `neurotwin` and `model` aliases can resolve to the same implementation and seed, creating duplicate ranking rows.
- Mamba is explicitly unavailable and raises an error.
- Several broad multimodal and clinical directions exist only in docs, configs, or synthetic fixtures.
- The A100 evidence bundle references checkpoints that are not included in the release asset.

## Inventory Verdict

Kahlus is a substantial research-engineering repository with real provenance, split, test, and evidence-gating work. Its main weakness is not missing infrastructure. It is that the central empirical task and the named architecture do not match the strongest manuscript interpretation.
