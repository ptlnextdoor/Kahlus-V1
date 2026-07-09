# Recent NeurIPS neuro/brain code-link reality check

Date: 2026-07-08

Purpose: check whether recent NeurIPS neuro, EEG, brain, fMRI, seizure, physiological-signal, and neural-dynamics papers expose public GitHub repositories that show how their figures are generated.

## What was scanned

- NeurIPS 2024 proceedings index.
- NeurIPS 2025 proceedings index.
- Brain/neuro/EEG-relevant title filter.
- Paper pages for the top curated subset.
- GitHub API fallback search for title + `github` when the paper page did not expose a GitHub link.

Artifacts:

- `neurips_neuro_title_harvest.json`: broad keyword harvest. This includes many false positives because “neural network” appears everywhere.
- `neurips_brain_relevant_harvest.json`: narrowed brain/neuro/EEG/physiology subset.
- `neurips_recent_code_link_audit.json`: paper-page plus GitHub API code-link audit for 66 high-priority recent papers.

## Result

The 2025 proceedings pages mostly do **not** expose GitHub links yet. GitHub API fallback search also found no credible official repos for many of the newest 2025 papers at scan time. This is expected for very recent NeurIPS papers: code often appears later through project pages, author repos, or camera-ready supplement updates.

This means Kahlus should not wait for 2025 repos before fixing figures. The strongest inspectable and reusable workflow remains:

1. **CEBRA**: separate figure repo with cached HDF5/CSV artifacts, Jupytext figure scripts, rendered notebooks, and CI.
2. **sEEGificant**: notebook-based signal heatmaps with behavior overlays.
3. **PopulationTransformer**: waveform plus spectrogram panel pattern.
4. **EEGPT / LaBraM / REVE**: useful for model/task organization and architecture language, but not public paper-figure scripts.

## How Kahlus now matches the usable pattern

Kahlus now has a CEBRA-style packet:

```text
docs/research/eeg_v1_figure_source/
├── data/
│   ├── task_results.csv
│   ├── baseline_ranking.csv
│   ├── audits.csv
│   ├── inventory.json
│   └── provenance.json
├── src/
│   ├── Figure1_eeg_v1_benchmark_suite.py
│   └── Figure2_eeg_v1_prediction_artifact_contract.py
└── figures/
    ├── Figure1_eeg_v1_benchmark_suite.{png,pdf,svg}
    └── Figure2_eeg_v1_prediction_artifact_contract.{png,pdf,svg}
```

This is the important matching move: figures are no longer generated from ad hoc drawing code alone. They are rendered from cached evidence artifacts by checked-in source scripts, and docs link figure → source → data.

## Next repos to re-check later

Re-check these after authors publish code:

- POSSM / hybrid SSM neural decoding: project page says `Code coming soon!`.
- BrainGPT: arXiv says code and models will be released.
- CSBrain: no code URL found in PDF/arXiv at scan time.
- 2025 EEG/brain foundation model papers in `neurips_recent_code_link_audit.json` with empty `github_links` and empty/highly uncertain search results.
