# Kahlus / NeuroTwin research documentation

Kahlus is a research codebase for leakage-safe neural translation benchmarks and NeuroTwin/NFC model development. These docs are organized like a scientific Python project: tutorials, methods, evidence, figures, API references, and paper-facing notes.

```{admonition} Current scientific posture
:class: important
The repo should not claim a new brain foundation model or clinical digital twin. The defensible target is careful neural translation, missing-modality reconstruction, future-state forecasting, and subject/site generalization under leakage-safe splits.
```

## What to read first

- [Concepts for non-specialists](overview/concepts.md): the plain-English version before the math.
- [Limitations and claim boundaries](limitations.md): what Kahlus is not, and unsafe claims to never make.
- [Ridge EEG diagnostics for Amrith](figures/ridge-eeg-diagnostics.md): benchmark-derived figure and analysis of why ridge regression on BNCI2014_001 measures input/target overlap, not learned neural dynamics.
- [Visual standards](figures/visual-standards.md): what counts as reputable neuroscience visualization vs AI-generated slop.
- [Reputable neuroscience repo patterns](references/reputable-neuroscience-repos.md): what MNE, MOABB, Braindecode, CEBRA, NeuroML, and Nilearn teach us.
- [Benchmark protocol](paper/benchmark_protocol.md): existing reviewer-facing benchmark rules.
- [Leakage-proof evaluation](methods/leakage-proof-evaluation.md): split and audit logic.

## Documentation map

```{toctree}
:maxdepth: 2
:caption: Overview

overview/concepts
limitations
```

```{toctree}
:maxdepth: 2
:caption: Research figures

figures/ridge-eeg-diagnostics
figures/visual-standards
analysis/ridge_eeg_interpretability_plan
analysis/neuroscience_figure_stack
```

```{toctree}
:maxdepth: 2
:caption: Methods and evidence

methods/leakage-proof-evaluation
methods/ridge-baseline
paper/benchmark_protocol
paper/methods
paper/experiments
paper/limitations
evidence/moabb_paper_audit_v1
results/a100-run-history
```

```{toctree}
:maxdepth: 2
:caption: Research design

research/neurotwin_project_state
research/neurotwin_master_research_state
research/neurotwin_nfc_mathematical_constitution
research/equation_ledger
research/falsification_core
research/pair_operator_design
```

```{toctree}
:maxdepth: 2
:caption: Runbooks and repo operation

DATA
LEAKAGE
BASELINES
CLAIMS
REPRODUCIBILITY
A100_RUNBOOK
ROADMAP
```

```{toctree}
:maxdepth: 2
:caption: External standards

references/reputable-neuroscience-repos
licenses/upstream-registry
```


```{toctree}
:maxdepth: 1
:caption: Appendix and archived notes

CHAPMAN_A100_LAUNCH
CHAPMAN_A100_QUICKSTART
H100_RUNBOOK
LICENSE_REUSE
RUNPOD_A100_REHEARSAL
analysis/ridge_eeg_figures/README
research/eeg_v1_figure_source/README
maintenance/repo-knowledge-graph
paper/abstract
paper/outline
paper/related_work
research/code_architecture_map
research/fnirs_observation_operator_notes
research/math_implementation_coverage
research/neurotwin-technical-report
research/nfc_architecture_decision_log
research/nfc_falsification_criteria
research/nfc_implementation_plan
research/pair_operator_readiness_report
research/repo_alignment_plan
research/turboquant_retrieval_notes
roadmap/coding_sprint_0_14_days
roadmap/kahlus_implementation_status
roadmap/sprint_ledger
research/kahlus_affect_researchdock_roadmap
research/kahlus_current_math_cs_report
research/kahlus_neural_casp_general_plan
research/kahlus_neurovisual_dataset_registry
research/kahlus_neurovisual_epilepsy_roadmap
research/kahlus_neurovisual_symptom_ontology
research/kahlus_pathway_matrix
research/kahlus_root_technical_spec
research/kahlus_stf_benchmark_math_note
research/kahlus_stf_public_dataset_review
research/m4_sleep_edf_preanalysis_plan
research/overnight_log
research/overnight_proposals
research/overnight_result
roadmap/kahlus_biomedical_execution_plan
roadmap/kahlus_thread_sprint_summary
roadmap/kahlus_v1_eeg_baseline_plan
roadmap/kahlus_v1_fewshot_adaptation_plan
```
