# Math Implementation Coverage

This matrix maps `docs/research/equation_ledger.md` to the current runnable repository. It is intentionally conservative: documentation coverage does not mean implementation, implementation does not mean validation, and synthetic artifacts do not create model-superiority claims.

## Coverage Key

- `implemented`: code path exists for the equation or a simplified equivalent.
- `tested`: there is a local unit/CLI/artifact test that exercises the code path.
- `used in experiment`: the equation contributes to a runnable suite or emitted artifact.
- `scaffolded`: a partial interface or proxy exists, but not the full mathematical object.
- `docs only`: preserved in research docs, with no runnable code path.
- `deferred`: explicitly future work.
- `archived`: no longer a main architecture claim.
- `falsified / not used`: invalidated or intentionally excluded from current claims.

## Summary

| Area | Current implementation coverage | Current tests/artifacts | Remaining gap |
| --- | --- | --- | --- |
| Core NFC equations | `EQ-003`, `EQ-004`, `EQ-006`, `EQ-009`, `EQ-010`, `EQ-011`, `EQ-012`, `EQ-017`, `EQ-018`, `EQ-019`, `EQ-020`, `EQ-057`, `EQ-058`, `EQ-079`, `EQ-084`, `EQ-085` are implemented as simplified experimental code. | `tests/models/test_nfc.py`, `tests/data/test_synthetic_field.py`, `tests/benchmarks/test_nfc_suite.py`, `nfc_synthetic_results.json`, `nfc_ablation_table.csv`, `evidence_gate.json`. | The field is not validated on real held-out neural data, and several equations are simplified neural modules rather than physics-complete operators. |
| Observation operators | fMRI, EEG, and behavior operators exist in `src/neurotwin/models/nfc/observations/`. | Shape/finite tests in `tests/models/test_nfc.py`; synthetic suite tasks in `tests/benchmarks/test_nfc_suite.py`. | fMRI HRF is a learned short convolution, EEG is pooled readout, behavior is shape-only, and MEG/spike/calcium/fNIRS are not implemented. |
| Synthetic proving ground | Synthetic latent field, fMRI-like observation, EEG-like observation, latent recovery, forecasting, masked reconstruction, and cross-modal tasks exist. | `tests/data/test_synthetic_field.py`; `tests/benchmarks/test_nfc_suite.py`; CLI emits `nfc_synthetic_results.json`, `nfc_synthetic_results.csv`, `nfc_ablation_table.csv`, `uncertainty_calibration.csv`, `evidence_gate.json`, `diagnostic_report.md`. | Synthetic generator lacks explicit physiology fields, artifact processes, true optical/fNIRS physics, and strict real-data validation. |
| Evidence gates | Leakage/claim gates, finalization, model cards, paper-mode gates, and NFC synthetic `--require-pass` behavior exist. | `tests/cli/test_expanded.py`, `tests/cli/test_report.py`, `tests/artifacts/test_docs_contracts.py`, `tests/training/test_prepared.py`. | Claims remain disabled unless final evidence gates pass; synthetic-only gates are plumbing/falsification checks. |
| Track A reproducibility | Split manifests, leakage audits, identity probes, prepared tasks, baselines, reports, and model cards are implemented. | Prepared-data and leakage tests under `tests/data/`, `tests/eval/`, `tests/benchmarks/`, `tests/training/`. | This track supports reproducibility claims, not architecture superiority. |
| fNIRS theory | `EQ-052` to `EQ-056` are documented in `docs/research/fnirs_observation_operator_notes.md`. | Documentation contract only. | No fNIRS adapter, optical forward model, SNIRF/BIDS fNIRS pipeline, MDD task, or clinical claim. |
| TurboQuant/TurboVec theory | `EQ-059` to `EQ-077` are documented in `docs/research/turboquant_retrieval_notes.md`. | Documentation contract only. | No retrieval package, no vector store, no semantic duplicate audit, no retrieval-kNN baseline. |
| Archived Pair-Operator math | Low-rank pair-state math is still implemented and tested, but Pair-Operator is no longer the main primitive. | `tests/models/test_nfc.py`, `tests/models/test_shapes.py`, `tests/benchmarks/test_nfc_suite.py`, `nfc_ablation_table.csv`. | Treat Pair-Operator as ablation/baseline only. |
| Future math only | Variational inference, gauge fixing, graph regularizers, stability penalties, InfoNCE, Koopman residuals, Riemannian EEG, sampling-lattice stress tests, sparse experts, and retrieval math are not runnable. | Research docs only. | Needs explicit implementation plans, tests, configs, and evidence gates before any claim. |

## Per-Equation Matrix

| ID | Coverage classification | Implementation path | Test path | Runnable config / artifact | Missing work or boundary |
| --- | --- | --- | --- | --- | --- |
| EQ-001 | implemented; tested; used as baseline | `src/neurotwin/models/torch_models.py`; `src/neurotwin/benchmarks/nfc_suite.py` | `tests/models/test_shapes.py`; `tests/benchmarks/test_nfc_suite.py` | `nfc_synthetic_results.json` model row `transformer` | Baseline only, not NFC novelty. |
| EQ-002 | docs only; deferred | none for input-selective Mamba; `TinySSMBaseline` is only a local fallback in `src/neurotwin/models/torch_models.py` | `tests/models/test_shapes.py`; runtime estimate tests mention `ssm_fallback` | `nfc_synthetic_results.json` model row `ssm_fallback` | No selective input-dependent Mamba/SSM backbone. |
| EQ-003 | implemented; tested; used in experiment | `src/neurotwin/models/nfc/latent_field.py`; `src/neurotwin/models/nfc/compiler.py` | `tests/models/test_nfc.py` | `nfc_synthetic_results.json`; `diagnostic_report.md` | Latent coordinates are experimental and not semantically identifiable. |
| EQ-004 | implemented simplified; tested; used in experiment | `src/neurotwin/models/nfc/compiler.py`; `src/neurotwin/models/nfc/observations/` | `tests/models/test_nfc.py`; `tests/benchmarks/test_nfc_suite.py` | `nfc_synthetic_results.json`; `evidence_gate.json` | Operators are learned/simple approximations, not full measurement physics. |
| EQ-005 | docs only; deferred | none | none | none | No probabilistic marginalization or latent posterior training. |
| EQ-006 | implemented simplified; tested | `src/neurotwin/models/nfc/field_update.py` | `tests/models/test_nfc.py` | NFC model path in synthetic suite | Uses discrete neural module, not a continuous controlled flow. |
| EQ-007 | docs only; deferred | none | none | none | No Neural CDE or irregular-time controlled differential equation path. |
| EQ-008 | docs only; deferred | none | none | none | No integro-differential neural-field solver. |
| EQ-009 | implemented simplified; tested | `src/neurotwin/models/nfc/field_update.py`; `src/neurotwin/data/synthetic_field.py` | `tests/models/test_nfc.py`; `tests/data/test_synthetic_field.py` | synthetic field sample; `nfc_synthetic_results.json` | Current update is not the full `D`, `K_t`, `B`, `xi_t` equation. |
| EQ-010 | implemented; tested; used in ablation | `src/neurotwin/models/nfc/pair_kernel.py`; `src/neurotwin/models/pair_operator.py` | `tests/models/test_nfc.py`; `tests/models/test_shapes.py` | `nfc_ablation_table.csv` | Implemented as low-rank learned kernel; real anatomical interpretation is unproven. |
| EQ-011 | implemented; tested; used in ablation | `src/neurotwin/models/nfc/pair_kernel.py` | `tests/models/test_nfc.py` | `nfc_ablation_table.csv` | Structural prior support exists but is not validated as anatomy. |
| EQ-012 | implemented; tested; used in ablation | `src/neurotwin/models/nfc/pair_kernel.py`; `src/neurotwin/models/nfc/field_update.py` | `tests/models/test_nfc.py` | `nfc_ablation_table.csv` | Pair update is ablation support, not a standalone claim. |
| EQ-013 | scaffolded; tested proxy | `src/neurotwin/models/nfc/uncertainty.py` | `tests/models/test_nfc.py`; `tests/benchmarks/test_nfc_suite.py` | `uncertainty_calibration.csv` | Pair uncertainty is derived from region uncertainty; no true `K_ij` error target. |
| EQ-014 | docs only; deferred | none | none | none | No graph-gradient operator. |
| EQ-015 | docs only; deferred | none | none | none | No explicit Laplacian construction in NFC loss. |
| EQ-016 | docs only; deferred | none | none | none | No graph smoothness regularizer. |
| EQ-017 | implemented simplified; tested | `src/neurotwin/models/nfc/observations/fmri.py`; `src/neurotwin/data/synthetic_field.py` | `tests/models/test_nfc.py`; `tests/data/test_synthetic_field.py` | synthetic fMRI output; `nfc_synthetic_results.json` | Uses short learned convolution/lag, not canonical HRF integral. |
| EQ-018 | implemented simplified; tested; used in experiment | `src/neurotwin/models/nfc/observations/fmri.py`; `src/neurotwin/benchmarks/nfc_suite.py` | `tests/models/test_nfc.py`; `tests/benchmarks/test_nfc_suite.py` | fMRI tasks in `nfc_synthetic_results.json` | Parcel readout is simplified and synthetic-only. |
| EQ-019 | implemented simplified; tested | `src/neurotwin/models/nfc/stimulus.py` | `tests/models/test_nfc.py` | NFC synthetic stimulus tasks | Causal averaging exists; no learned HRF-grade stimulus alignment. |
| EQ-020 | implemented simplified; tested; used in experiment | `src/neurotwin/models/nfc/observations/eeg.py`; `src/neurotwin/data/synthetic_field.py` | `tests/models/test_nfc.py`; `tests/data/test_synthetic_field.py`; `tests/benchmarks/test_nfc_suite.py` | EEG tasks in `nfc_synthetic_results.json` | No real lead-field matrix or source-current model. |
| EQ-021 | docs only; deferred | none | none | none | No MEG adapter or claim. |
| EQ-022 | docs only; deferred | none | none | none | No spike observation operator. |
| EQ-023 | docs only; deferred | none | none | none | No calcium observation operator. |
| EQ-024 | scaffolded; tested | `src/neurotwin/models/nfc/observations/behavior.py`; `src/neurotwin/models/nfc/compiler.py` | `tests/models/test_nfc.py` | no current benchmark artifact | Behavior head is shape-tested only; no behavior task claim. |
| EQ-025 | docs only; deferred | none | none | none | No variational posterior. |
| EQ-026 | docs only; deferred | none | none | none | No ELBO objective. |
| EQ-027 | docs only | none | none | docs only | Gauge ambiguity is documented but not enforced. |
| EQ-028 | docs only | none | none | docs only | Gauge ambiguity is documented but not enforced. |
| EQ-029 | docs only | none | none | docs only | Gauge ambiguity is documented but not enforced. |
| EQ-030 | docs only; deferred | none | none | none | No temporal smoothness loss. |
| EQ-031 | docs only; deferred | none | none | none | No spectral regularizer. |
| EQ-032 | scaffolded elsewhere; not NFC-proven | prepared/translator subject controls in `src/neurotwin/models/torch_models.py`; config plumbing in training | `tests/models/test_shapes.py`; `tests/training/test_prepared.py` | prepared-training artifacts when enabled | No NFC low-rank subject adapter of this exact form. |
| EQ-033 | docs only; deferred | none | none | none | No continuous-time stability model. |
| EQ-034 | docs only; deferred | none | none | none | No Jacobian audit. |
| EQ-035 | docs only; deferred | none | none | none | No eigenvalue stability criterion. |
| EQ-036 | docs only; deferred | none | none | none | No spectral-radius enforcement. |
| EQ-037 | docs only; deferred | none | none | none | No stability penalty. |
| EQ-038 | docs only; scaffolded proxy | `src/neurotwin/models/nfc/uncertainty.py` emits positive maps | `tests/models/test_nfc.py` | `uncertainty_calibration.csv` proxy | No NLL training objective or predictive variance likelihood. |
| EQ-039 | scaffolded proxy; needs evidence | `src/neurotwin/benchmarks/nfc_suite.py` writes calibration proxy | `tests/benchmarks/test_nfc_suite.py` | `uncertainty_calibration.csv` with `proxy_source=mse_derived_suite_proxy` | No interval coverage target or real calibration claim. |
| EQ-040 | docs only; deferred | none | none | none | No InfoNCE objective or leakage-safe contrastive sampler. |
| EQ-041 | docs only; deferred | none | none | none | No Koopman observable model. |
| EQ-042 | docs only; deferred | none | none | none | No Koopman residual update. |
| EQ-043 | docs only; deferred | none | none | none | No Koopman residual loss or spectral penalty. |
| EQ-044 | docs only; deferred | none | none | none | No EEG covariance baseline/head. |
| EQ-045 | docs only; deferred | none | none | none | No SPD covariance output head. |
| EQ-046 | docs only; deferred | none | none | none | No AIRM metric. |
| EQ-047 | docs only; deferred | existing manifests store sample rates/times but not this lattice model | split/audit tests cover leakage, not lattice math | none | No native-lattice multimodal timing model. |
| EQ-048 | docs only; deferred | none | none | none | No union-of-grids operator. |
| EQ-049 | docs only; deferred | none | none | none | No Fourier feature module for NFC timing. |
| EQ-050 | docs only; deferred | none | none | none | No graph spectral basis. |
| EQ-051 | docs only; deferred | none | none | none | No coprime window stress-test generator. |
| EQ-052 | docs only; deferred | `docs/research/fnirs_observation_operator_notes.md` | documentation contract only | docs only | No fNIRS optical operator. |
| EQ-053 | docs only; deferred | `docs/research/fnirs_observation_operator_notes.md` | documentation contract only | docs only | No Rytov transform code. |
| EQ-054 | docs only; deferred | `docs/research/fnirs_observation_operator_notes.md` | documentation contract only | docs only | No optical density output path. |
| EQ-055 | docs only; deferred | `docs/research/fnirs_observation_operator_notes.md` | documentation contract only | docs only | No fNIRS observation support or claim. |
| EQ-056 | docs only; deferred | `docs/research/fnirs_observation_operator_notes.md` | documentation contract only | docs only | Synthetic generator has no explicit hemodynamic/physiology field. |
| EQ-057 | implemented simplified; tested; used in experiment | `src/neurotwin/data/synthetic_field.py`; `src/neurotwin/benchmarks/nfc_suite.py` | `tests/data/test_synthetic_field.py`; `tests/benchmarks/test_nfc_suite.py` | `nfc_synthetic_results.json` fMRI rows | Simplified lag/readout, not a full hemodynamic field. |
| EQ-058 | implemented simplified; tested; used in experiment | `src/neurotwin/data/synthetic_field.py`; `src/neurotwin/benchmarks/nfc_suite.py` | `tests/data/test_synthetic_field.py`; `tests/benchmarks/test_nfc_suite.py` | `nfc_synthetic_results.json` EEG rows | No explicit motion artifact process. |
| EQ-059 | docs only; deferred | `docs/research/turboquant_retrieval_notes.md` | documentation contract only | docs only | No quantization API. |
| EQ-060 | docs only; deferred | `docs/research/turboquant_retrieval_notes.md` | documentation contract only | docs only | No MSE compression objective. |
| EQ-061 | docs only; deferred | `docs/research/turboquant_retrieval_notes.md` | documentation contract only | docs only | No inner-product quantization objective. |
| EQ-062 | docs only; deferred | `docs/research/turboquant_retrieval_notes.md` | documentation contract only | docs only | No vector-store normalization layer. |
| EQ-063 | docs only; deferred | `docs/research/turboquant_retrieval_notes.md` | documentation contract only | docs only | No random rotation quantizer. |
| EQ-064 | docs only; deferred | `docs/research/turboquant_retrieval_notes.md` | documentation contract only | docs only | No TurboQuant implementation. |
| EQ-065 | docs only; deferred | `docs/research/turboquant_retrieval_notes.md` | documentation contract only | docs only | No dimension/risk audit. |
| EQ-066 | docs only; deferred | `docs/research/turboquant_retrieval_notes.md` | documentation contract only | docs only | No scalar Lloyd-Max buckets. |
| EQ-067 | docs only; deferred | `docs/research/turboquant_retrieval_notes.md` | documentation contract only | docs only | No residual compression. |
| EQ-068 | docs only; deferred | `docs/research/turboquant_retrieval_notes.md` | documentation contract only | docs only | No QJL residual correction. |
| EQ-069 | docs only; deferred | `docs/research/turboquant_retrieval_notes.md` | documentation contract only | docs only | No retrieval scoring backend. |
| EQ-070 | docs only; deferred | `docs/research/turboquant_retrieval_notes.md` | documentation contract only | docs only | No stimulus vector index. |
| EQ-071 | docs only; deferred | `docs/research/turboquant_retrieval_notes.md` | documentation contract only | docs only | No latent field memory bank. |
| EQ-072 | docs only; deferred | `docs/research/turboquant_retrieval_notes.md` | documentation contract only | docs only | No network-wise latent memory. |
| EQ-073 | docs only; deferred | `docs/research/turboquant_retrieval_notes.md` | documentation contract only | docs only | No retrieval neighborhood API. |
| EQ-074 | docs only; deferred | `docs/research/turboquant_retrieval_notes.md` | documentation contract only | docs only | No retrieval-kNN baseline. |
| EQ-075 | docs only; deferred | `docs/research/turboquant_retrieval_notes.md` | documentation contract only | docs only | No retrieval weighting function. |
| EQ-076 | docs only; deferred | `docs/research/turboquant_retrieval_notes.md` | documentation contract only | docs only | No quantized-vs-exact recall/distortion audit. |
| EQ-077 | docs only; deferred | `docs/research/turboquant_retrieval_notes.md` | documentation contract only | docs only | No semantic near-duplicate audit. |
| EQ-078 | docs only; deferred | partial terms exist across NFC and prepared training, but no single full NFC loss | tests cover partial pieces | none | No integrated latent/graph/stability/uncertainty/calibration loss. |
| EQ-079 | implemented; tested; used in experiment | `src/neurotwin/scoring/metrics.py`; `src/neurotwin/benchmarks/nfc_suite.py` | `tests/scoring/test_metrics.py`; `tests/benchmarks/test_nfc_suite.py` | `nfc_synthetic_results.json`; `nfc_synthetic_results.csv` | Synthetic MSE only; no real fMRI claim. |
| EQ-080 | partially implemented; docs-only PSD term | `src/neurotwin/scoring/metrics.py`; `src/neurotwin/benchmarks/nfc_suite.py` | `tests/scoring/test_metrics.py`; `tests/benchmarks/test_nfc_suite.py` | EEG MSE rows in `nfc_synthetic_results.json` | PSD loss term is not implemented. |
| EQ-081 | docs only; deferred | none | none | none | No spike likelihood. |
| EQ-082 | experimental framing; partially used in synthetic tasks | `src/neurotwin/benchmarks/nfc_suite.py`; `src/neurotwin/models/nfc/compiler.py` | `tests/benchmarks/test_nfc_suite.py`; `tests/models/test_nfc.py` | `synthetic_eeg_to_fmri`; `synthetic_fmri_to_eeg` in `nfc_synthetic_results.json` | No explicit separate inference operator `I_theta`; current path is model forward task. |
| EQ-083 | docs only; research framing | `docs/research/neurotwin_master_research_state.md` | none | docs only | No formal commutative-diagram API. |
| EQ-084 | implemented; tested; used as baseline | `src/neurotwin/benchmarks/nfc_suite.py`; `src/neurotwin/models/baselines.py`; `src/neurotwin/models/torch_models.py` | `tests/benchmarks/test_nfc_suite.py`; `tests/benchmarks/test_baseline_suite.py` | direct baseline rows in `nfc_synthetic_results.json` | Direct translation is the baseline lane, not the main claim. |
| EQ-085 | implemented experimental; tested; used in experiment | `src/neurotwin/models/nfc/compiler.py`; `src/neurotwin/benchmarks/nfc_suite.py` | `tests/models/test_nfc.py`; `tests/benchmarks/test_nfc_suite.py` | `nfc_full` rows in `nfc_synthetic_results.json`; `nfc_ablation_table.csv` | Needs strict synthetic pass and real held-out evidence before any architecture claim. |

## What Is Actually Runnable Today

The repo can currently run the following math-backed surfaces locally:

```bash
PYTHONPATH=src python3 -m neurotwin.cli eval \
  --suite nfc_synthetic \
  --out-dir /tmp/neurotwin_math_coverage_nfc \
  --train-steps 1 \
  --seed 0
```

Expected outputs:

- `nfc_synthetic_results.json`
- `nfc_synthetic_results.csv`
- `nfc_ablation_table.csv`
- `nfc_falsification_report.md`
- `uncertainty_calibration.csv`
- `evidence_gate.json`
- `diagnostic_report.md`

These artifacts prove the suite compiles and the gate can inspect the required tasks/models. They do not prove NeuroTwin/NFC superiority.

## Coverage Conclusions

1. The runnable core is a real NFC skeleton: latent field, field update, pair-kernel ablation, simplified observation operators, uncertainty maps, synthetic field generator, and strict synthetic gate.
2. The repo has not implemented the full mathematical vision. Variational inference, graph gauge fixing, full stability theory, InfoNCE, Koopman residuals, Riemannian EEG, fNIRS optical physics, TurboVec retrieval, semantic duplicate audit, sparse experts, and real Algonauts/CNeuroMod evidence are still deferred or docs-only.
3. Pair-Operator is implemented and tested, but only as a baseline/ablation. It is archived as the main primitive.
4. fNIRS and TurboQuant/TurboVec are useful research directions, but they are not current capabilities.
5. The next evidence-bearing step remains the strict NFC synthetic diagnostic. A100/Algonauts/6x DDP should not proceed until the synthetic gate produces a real pass.
