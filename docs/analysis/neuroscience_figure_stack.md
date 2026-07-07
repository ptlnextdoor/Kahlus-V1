# Neuroscience Figure Stack for Kahlus

This note records the recommended plotting stack for Kahlus EEG/ridge diagnostics and future neuroscience figures. The goal is to avoid prompt-generated fake diagrams and instead generate figures from benchmark tensors with publication-style defaults.

## Core recommendation

Use **data-driven Python figures**, not hand-prompted SVG/JPG generation.

Recommended stack:

1. **MNE-Python** for EEG/MEG/ECoG neurophysiology visualization
   - Raw traces, epochs, evoked responses, topomaps, PSDs, time-frequency plots, sensor layouts.
   - Standard in EEG papers and BCI tooling.
   - GitHub: `mne-tools/mne-python`
   - Docs include a dedicated “Make figures more publication ready” tutorial.

2. **MOABB** for EEG benchmark conventions
   - Mother of All BCI Benchmarks.
   - Built on MNE-Python and scikit-learn.
   - Use as reference for reproducible EEG benchmark framing, dataset handling, and evaluation methodology.
   - GitHub: `NeuroTechX/moabb`

3. **Braindecode** for deep-learning EEG paper conventions
   - PyTorch toolbox for raw EEG/ECoG/MEG decoding.
   - Good reference for model result plots, EEGNet-style context, and deep EEG benchmark presentation.
   - GitHub: `braindecode/braindecode`

4. **Matplotlib + SciencePlots** for final publication styling
   - Matplotlib remains the base layer for paper figures.
   - SciencePlots provides publication-style rcParams for papers/theses.
   - Export both `.png` at 300+ DPI and `.pdf`/`.svg` vector versions.
   - GitHub: `garrettj403/SciencePlots`

5. **Seaborn / scipy / scikit-learn** for statistics and diagnostics
   - Use for confidence intervals, distributions, correlation heatmaps, residual plots, and model metric summaries.
   - Keep plot rendering in Matplotlib so final layout is controlled.

6. **Nilearn** only if Kahlus shifts to fMRI/brain-volume figures
   - Glass brain, surface maps, ROI/connectivity visualizations, statistical maps.
   - Not needed for current EEG ridge waveform diagnostics.

7. **pyNeuroML / libNeuroML** only for NeuroML cell/network morphology and electrophysiology
   - NeuroML docs emphasize morphology plots, interactive 3D cell visualization, F-I curves, voltage traces.
   - Useful for NeuroML model documentation, not the current MOABB EEG ridge benchmark.

## What the current Amrith request needs

Amrith asked for analysis of the existing benchmark, not more benchmarks. For the EEG ridge baseline, the first figure packet should include:

1. Raw EEG waveforms with the current window `X_t` and next-window target `X_{t+1}` labeled.
2. The actual ridge design matrix shown as `[windows × time, channels]`, matching the repo's current `linear_ridge` implementation.
3. Ridge predicted vs actual future EEG traces, with residuals and per-channel metrics.
4. Autocorrelation / lag diagnostics explaining why a linear model can be strong.
5. Ridge coefficient maps reshaped back into input-channel/output-channel or input-time/output-time structure.
6. Optional MNE topomap if real channel names/positions are available.

## Visual grammar

- White background.
- No decorative gradients.
- Use muted scientific colors:
  - input/current window: blue
  - target/future window: red/orange
  - actual signal: near-black
  - ridge prediction: red
  - residual/error: gray
- Label panels as A, B, C, D when composing multi-panel figures.
- Every schematic must be stamped `SCHEMATIC, NOT BENCHMARK EVIDENCE`.
- Every benchmark figure must include data source, split, model, and metric in the caption or README.
- Avoid dense overlapping lines. Show 4-8 channels max per waveform panel.
- Export:
  - `fig*.png` at 300 DPI for slides/email
  - `fig*.pdf` for paper/docs

## Implementation path

Current script:

```bash
python3 scripts/analysis/plot_ridge_eeg_diagnostics.py
```

This generates schematic layout figures only.

For real benchmark evidence, create an `.npz` containing:

- `x_train`: `[windows, time, channels]`
- `y_train`: `[windows, time, channels]`
- `x_test`: `[windows, time, channels]`
- `y_test`: `[windows, time, channels]`
- optional `y_pred_test`: exported ridge predictions from the benchmark run
- optional `sfreq`
- optional `channel_names`

Then run:

```bash
python3 scripts/analysis/plot_ridge_eeg_diagnostics.py \
  --npz path/to/real_benchmark_tensors.npz \
  --time-length 128 \
  --n-channels <N>
```

If real channel names and montage are available, extend the script with MNE:

```python
import mne
info = mne.create_info(channel_names, sfreq, ch_types="eeg")
info.set_montage("standard_1020", on_missing="ignore")
mne.viz.plot_topomap(values, info)
```

## Sources / reference projects

- MNE-Python: `https://github.com/mne-tools/mne-python`
- MNE publication-ready figures tutorial: `https://mne.tools/stable/auto_tutorials/visualization/10_publication_figure.html`
- MNE topomap examples: `https://mne.tools/stable/auto_examples/visualization/evoked_topomap.html`
- MOABB: `https://github.com/NeuroTechX/moabb`
- Braindecode: `https://github.com/braindecode/braindecode`
- Nilearn plotting: `https://nilearn.github.io/stable/plotting/index.html`
- NeuroML visualization docs: `https://docs.neuroml.org/Userdocs/VisualisingCells.html`
- SciencePlots: `https://github.com/garrettj403/SciencePlots`
