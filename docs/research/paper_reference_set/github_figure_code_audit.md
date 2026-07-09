# GitHub figure-code audit for the paper reference set

Date: 2026-07-08

Purpose: identify which reference papers ship usable code for their published figures, not just model training code. This matters because Kahlus should imitate the reproducible figure workflows, not copy screenshots or hand-draw diagrams.

## Executive finding

Most EEG foundation-model repositories do **not** ship exact paper-figure scripts. They usually provide training, preprocessing, and evaluation entrypoints, plus one static architecture image in the README. The strongest counterexample is **CEBRA**, which separates model code from a dedicated `cebra-figures` repository containing cached HDF5/CSV results, Jupytext Python notebooks, generated notebooks, and CI that renders the figures.

Kahlus should copy the CEBRA-style split:

```text
model/package repo
  -> training/eval code

figure-reference repo or docs/figures/source
  -> data/*.h5 or data/*.csv cached results
  -> src/Figure1.py, src/Figure2.py as Jupytext notebooks
  -> figures/Figure1.ipynb rendered notebooks
  -> CI: install deps, make all, make html
```

This is much more credible than generating standalone SVGs from vibes.

## Repo status table

| Paper key | Official code found? | Repo | Figure-code status | Usefulness for Kahlus |
|---|---:|---|---|---|
| `eegpt_neurips2024` | Yes | <https://github.com/BINE022/EEGPT> | Mostly training/eval. README embeds `figures/EEGPT.jpg`; no exact paper plotting scripts found. | Use for task/eval organization, not figure generation. |
| `seeg_electrode_variability_neurips2024` | Yes | <https://github.com/gmentz/seegnificant> | Has `example.ipynb` and `Signal_Processing/utils.py` with real sEEG plotting utilities. Does not appear to ship exact paper panel scripts. | Useful for high-gamma trial heatmaps, reaction-time overlays, and train/val/test metric plots. |
| `labram_iclr2024` | Yes | <https://github.com/935963004/LaBraM> | Mostly preprocessing, pretraining, finetuning. README embeds `labram.png`; no figure-generation scripts found. | Use for architecture naming and EEG preprocessing contract, not plotting. |
| `population_transformer_2024` | Yes | <https://github.com/czlwang/PopulationTransformer> | Has TensorBoard waveform/spectrogram helpers and a superlet demo plot, but no exact paper panels found. | Useful for spectrogram/time-frequency panel mechanics. |
| `hybrid_ssm_neural_decoding_2025` | Project page only | <https://possm-brain.github.io> | Project page says “Code coming soon!” No repo to inspect yet. | Re-check later. Use paper figures manually for style only. |
| `braingpt_2024` | Not yet | arXiv says “code and models will be released” | No GitHub URL in PDF/arXiv and no credible GitHub API hit. | Re-check later. |
| `reve_eeg_foundation_2025` | Yes | <https://github.com/elouayas/reve_eeg> | Training/eval/config repo. Project page has results tables. No exact figure scripts found in current repo. | Use for benchmark table structure and Hydra task organization. |
| `csbrain_2025` | Not found | arXiv/PDF only | No GitHub URL in PDF/arXiv and no credible GitHub API hit. | Re-check later. |
| `cebra_2022` | Yes | <https://github.com/AdaptiveMotorControlLab/CEBRA> and <https://github.com/AdaptiveMotorControlLab/cebra-figures> | Best-in-class. The `cebra-figures` repo ships cached result data, Jupytext figure scripts, rendered notebooks, and CI. | Copy this workflow almost directly for Kahlus figures. |

## Best source to imitate: CEBRA

### Model repo

- Repo: <https://github.com/AdaptiveMotorControlLab/CEBRA>
- Local audit checkout: `/tmp/kahlus-paper-repos-20260708/cebra_2022`
- Relevant files:
  - `docs/source/figures.rst`
  - `docs/Makefile`
  - `docs/source/conf.py`
  - `cebra/integrations/matplotlib.py`
  - `cebra/integrations/plotly.py`

The docs explicitly say they provide:

1. Demo notebooks for common use cases.
2. Plotting code for all paper figures, generated from cached experimental results.
3. Experiment collections for recomputing the results on a cluster.

`docs/Makefile` pulls figure sources from a separate repo:

```text
git clone --depth 1 git@github.com:AdaptiveMotorControlLab/cebra-figures.git source/cebra-figures
git clone --depth 1 git@github.com:AdaptiveMotorControlLab/cebra-demos.git source/demo_notebooks
```

### Dedicated figure repo

- Repo: <https://github.com/AdaptiveMotorControlLab/cebra-figures>
- Local audit checkout: `/tmp/kahlus-paper-repos-20260708/cebra_figures_source`
- Commit inspected: `42e9420`

Key layout:

```text
cebra-figures/
├── data/
│   ├── Figure1.h5
│   ├── Figure2.h5
│   ├── Figure3.h5
│   ├── results_v3/*.csv
│   └── ...
├── src/
│   ├── Figure1.py
│   ├── Figure2.py
│   ├── Figure3.py
│   ├── Figure4.py
│   ├── Figure5.py
│   └── ExtendedDataFigure*.py
├── figures/
│   ├── Figure1.ipynb
│   ├── Figure2.ipynb
│   └── ExtendedDataFigure*.ipynb
└── .github/workflows/figures.yml
```

Important README statement:

> This repo only contains plotting functions which can be applied to pre-computed results.

This is exactly the missing Kahlus pattern: do not mix figure scripts with training code, and do not require rerunning huge experiments just to render docs.

### What the CEBRA figure scripts actually do

`src/Figure1.py`:

- Loads cached results with `pd.read_hdf("../data/Figure1.h5", key="data")`.
- Uses `matplotlib`, `seaborn`, `pandas`, `numpy`, and `scipy.stats`.
- Plots tiny latent points with `ax.scatter(..., s=1, c=labels, cmap="cool")`.
- Uses stripplots for distributions: `sns.stripplot(..., color="black", s=3, jitter=0.15)`.
- Overlays median/summary points in orange.
- Exports individual panels as `png` and `svg` with `bbox_inches="tight"` and `transparent=True`.
- Prints ANOVA and Tukey HSD reports inline in the notebook, tying visuals to statistics.

`src/Figure2.py`:

- Loads cached HDF5 artifacts with `pd.read_hdf("../data/Figure2.h5", key="data")` and `pd.read_hdf("../data/SupplVideo1.h5", key="data")`.
- Uses 3D latent scatter panels for hypothesis/discovery embeddings.
- Plots loss curves with explicit labels: `Iterations`, `InfoNCE Loss`.
- Plots true versus predicted trajectories with physical units: `Time [s]`, `Position [cm]`.
- Uses side-by-side model comparison panels plus saved SVG panels.

`cebra/integrations/matplotlib.py`:

- Provides reusable plotting API instead of one-off scripts.
- Implements plotters for training temperature, loss, embeddings, and consistency matrices.
- Uses 2D/3D embedding plotting with labels mapped to color.
- Explicitly validates embedding dimensions and label lengths.

`cebra/integrations/plotly.py`:

- Mirrors embedding plots in Plotly for interactive web docs.
- Converts Matplotlib colormaps into Plotly colorscales.

### CEBRA pattern to copy into Kahlus

```text
docs/research/eeg_v1_figure_source/
├── data/
│   ├── ridge_benchmark_results.csv
│   ├── leakage_audit.json
│   ├── prediction_examples.npz
│   └── provenance.json
├── src/
│   ├── Figure1_pipeline.py
│   ├── Figure2_benchmark.py
│   ├── Figure3_leakage_diagnostics.py
│   └── Figure4_residual_psd.py
├── notebooks/
│   ├── Figure1_pipeline.ipynb
│   └── Figure2_benchmark.ipynb
└── Makefile
```

Rules:

1. Training scripts write compact cached artifacts: CSV, JSON, HDF5, NPZ.
2. Figure scripts only read cached artifacts and render panels.
3. Every public figure has a Jupytext `.py` source and rendered notebook.
4. Every figure exports SVG/PDF plus PNG.
5. Captions link to the source script and data artifact.
6. Docs can build without rerunning expensive training.

## sEEGificant patterns worth copying

Repo: <https://github.com/gmentz/seegnificant>

Relevant files:

- `example.ipynb`
- `Signal_Processing/utils.py`

Good pattern in `example.ipynb`:

```python
pcm = axs[j].pcolor(t, np.arange(x.shape[0]), x[:, :, i], vmin=0, vmax=1, cmap="cool")
axs[j].set(ylabel="Trial No.", xlabel="Time (sec)", title=f"High-γ | Elec {i}")
fig.colorbar(pcm, ax=axs[j])
axs[j].scatter((respRT_sorted) / 1000, np.arange(len(order)), label="RT", color="k")
```

Why it matters:

- It is a real neuroscience panel: trials by time, neural amplitude as color, behavior overlaid as black reaction-time dots.
- It uses event/task-aligned time windows.
- It makes electrode-level heterogeneity visible instead of hiding it.

Kahlus adaptation:

- For EEG windows, create a trial/time heatmap panel only when we have real arrays.
- Overlay event markers, prediction horizon, seizure/sleep labels, or residual spikes as black dots/lines.
- Use physiological labels and units, not generic “value”.

Weakness to avoid:

- The utility currently labels voltage as `Voltage (?)`. Kahlus should never use unknown units in public figures. If units are unknown, label the figure as internal diagnostic only.

## PopulationTransformer patterns worth copying

Repo: <https://github.com/czlwang/PopulationTransformer>

Relevant files:

- `util/tensorboard_utils.py`
- `preprocessors/superlet.py`

Useful pattern in `preprocessors/superlet.py`:

```python
fig, (ax1, ax2) = ppl.subplots(
    2,
    1,
    sharex=True,
    gridspec_kw={"height_ratios": [1, 3]},
    figsize=(6, 6),
)
ax1.plot(np.arange(signal.size) / fs, signal, c="cornflowerblue")
ax1.set_ylabel("signal (a.u.)")
im = ax2.imshow(ampls, cmap="magma", aspect="auto", extent=extent, origin="lower")
ppl.colorbar(im, ax=ax2, orientation="horizontal", shrink=0.7, pad=0.2, label="amplitude (a.u.)")
ax2.set_xlabel("time (s)")
ax2.set_ylabel("frequency (Hz)")
```

Why it matters:

- This is a clean time-frequency panel structure: waveform on top, spectrogram below, shared time axis.
- It uses `extent` so axes show seconds and Hz instead of pixel indices.

Kahlus adaptation:

- Use this exact structure for EEG PSD/time-frequency diagnostics.
- Replace `a.u.` with real units if known, or explicitly label as normalized amplitude.

## EEGPT, LaBraM, REVE: useful but not for figure code

### EEGPT

Repo: <https://github.com/BINE022/EEGPT>

- Official repo exists.
- README embeds `figures/EEGPT.jpg`.
- The repo provides dataset preparation, pretraining, linear probe, and fine-tuning scripts.
- No exact figure-generation scripts were found in the audit scan.

Kahlus takeaway:

- Use their organization of downstream tasks and pretrained-model references.
- Do not look here for publication-grade plotting workflow.

### LaBraM

Repo: <https://github.com/935963004/LaBraM>

- Official repo exists.
- README embeds `labram.png`.
- Provides dataset makers, HDF5 preprocessing, pretraining, vector-quantized neural spectrum prediction, and finetuning.
- No exact figure-generation scripts were found in the audit scan.

Kahlus takeaway:

- Use their preprocessing contract language: remove irrelevant channels, bandpass range, notch filter, resample rate, units.
- Do not imitate the repo as a docs/figure architecture.

### REVE

Repo: <https://github.com/elouayas/reve_eeg>

- Official repo exists and is linked from <https://brain-bzh.github.io/reve/>.
- It is a Hydra training/evaluation codebase with preprocessing and Hugging Face exports.
- No exact figure-generation scripts were found in the audit scan.
- The project page has benchmark tables and dataset-composition visuals, but not the source scripts in the cloned repo.

Kahlus takeaway:

- Use their benchmark table style and broad task taxonomy.
- Do not rely on the repo for paper panel generation.

## Unresolved or not-yet-released repos

### POSSM / hybrid SSM

Paper: `Generalizable, real-time neural decoding with hybrid state-space models`

Project page: <https://possm-brain.github.io>

Status: the project page currently says `Code coming soon!`. The PDF says the code will be released through `torch_brain` and linked through the project page.

### BrainGPT

Paper: `BrainGPT: Unleashing the Potential of EEG Generalist Foundation Model by Autoregressive Pre-training`

Status: arXiv says `The code and models will be released.` No GitHub URL was found in the PDF, arXiv page, or GitHub API search.

### CSBrain

Paper: `CSBrain: A Cross-scale Spatiotemporal Brain Foundation Model for EEG Decoding`

Status: no code URL was found in the PDF, arXiv page, or GitHub API search.

## Immediate Kahlus action items

1. Create a `docs/research/eeg_v1_figure_source/` or `examples/figures/eeg_v1/` spine modeled on `cebra-figures`.
2. Modify Kahlus training/eval jobs to export compact plot artifacts:
   - `task_results.csv`
   - `baseline_ranking.csv`
   - `leakage_report.json`
   - `prediction_examples.npz` with `x_test`, `y_test`, `y_pred`, `time_s`, `channel_names`
   - `provenance.json` with git SHA, command, seed, data path/hash
3. Rebuild the public figures from cached artifacts only.
4. Promote figures to public docs only when their artifact source exists.
5. Use CEBRA-style strips/scatters for evidence figures and sEEGificant-style trial heatmaps for signal diagnostics.

## Audit artifacts

- `github_url_extracts.json`: GitHub/code URLs extracted from the downloaded PDFs.
- `github_figure_code_scan.json`: local scan results for plotting terms across cloned repos.
- Temporary inspected clones: `/tmp/kahlus-paper-repos-20260708`.
