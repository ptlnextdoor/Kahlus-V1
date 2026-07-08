# Kahlus / NeuroTwin research docs

Kahlus is a research codebase for leakage-safe neural translation benchmarks and NeuroTwin/NFC model development. These docs are organized like a scientific Python project: tutorials, methods, evidence, figures, API references, and paper-facing notes.

```{admonition} Current scientific posture
:class: important
Kahlus should not claim a new brain foundation model, a clinical digital twin, or a clinical decision system. The defensible target is careful neural translation, missing-modality reconstruction, future-state forecasting, and subject/site/dataset generalization under leakage-safe splits.
```

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} Beginner path
:link: overview/concepts
:link-type: doc
What is EEG? What is a forecast? Why do leakage-safe splits matter? Start here if you are curious about brains but do not know the jargon yet.
:::

:::{grid-item-card} Methods path
:link: methods/ridge-baseline
:link-type: doc
Equations, implementation details, baselines, failure modes, and what the current ridge result does and does not prove.
:::

:::{grid-item-card} Reviewer path
:link: figures/eeg-v1-ridge-visuals
:link-type: doc
Figures, provenance, benchmark caveats, visual standards, and exact scripts used to render evidence artifacts.
:::

::::

## What to read first

- [EEG/ridge evidence figures from versions](figures/eeg-v1-ridge-visuals.md): artifact-first plots generated from saved CSV/JSON evidence bundles.
- [Ridge baseline evidence note](methods/ridge-baseline.md): beginner-to-expert explanation with equations, artifact boundaries, and caveats.
- [Visual standards](figures/visual-standards.md): what counts as reputable neuroscience visualization versus AI-generated slop.
- [Reputable neuroscience repo patterns](references/reputable-neuroscience-repos.md): what MNE, MOABB, Braindecode, CEBRA, NeuroML, and Nilearn teach us.
- [Limitations](limitations.md): explicit claim boundaries.

```{toctree}
:maxdepth: 2
:caption: Start here

overview/concepts
limitations
```

```{toctree}
:maxdepth: 2
:caption: Figure pages

figures/eeg-v1-ridge-visuals
```

```{toctree}
:maxdepth: 2
:caption: Standards and references

figures/visual-standards
references/reputable-neuroscience-repos
```

```{toctree}
:maxdepth: 2
:caption: Methods and evidence

methods/ridge-baseline
methods/leakage-proof-evaluation
paper/benchmark_protocol
paper/methods
paper/experiments
paper/limitations
evidence/moabb_paper_audit_v1
results/a100-run-history
```

```{toctree}
:maxdepth: 2
:caption: Executable examples

auto_examples/index
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

licenses/upstream-registry
```

```{toctree}
:maxdepth: 1
:caption: Appendix and archived notes

research/eeg_v1_ridge_visuals/eeg_v1_ridge_visual_analysis
CHAPMAN_A100_LAUNCH
CHAPMAN_A100_QUICKSTART
H100_RUNBOOK
LICENSE_REUSE
RUNPOD_A100_REHEARSAL
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
research/ridge_eeg_diagnostic_schematics/README
research/turboquant_retrieval_notes
research/kahlus_affect_researchdock_roadmap
research/kahlus_current_math_cs_report
research/kahlus_neurovisual_dataset_registry
research/kahlus_neurovisual_epilepsy_roadmap
research/kahlus_neurovisual_symptom_ontology
research/kahlus_pathway_matrix
research/kahlus_root_technical_spec
research/kahlus_stf_benchmark_math_note
research/kahlus_stf_public_dataset_review
roadmap/coding_sprint_0_14_days
roadmap/kahlus_biomedical_execution_plan
roadmap/kahlus_implementation_status
roadmap/kahlus_thread_sprint_summary
roadmap/kahlus_v1_eeg_baseline_plan
roadmap/kahlus_v1_fewshot_adaptation_plan
roadmap/sprint_ledger
```
