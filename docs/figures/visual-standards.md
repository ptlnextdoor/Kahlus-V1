# Visual standards: reputable neuroscience vs AI slop

The visual bar for Kahlus is: **every research figure must be generated from data, code, and provenance**. Prompted SVGs are allowed only for explicitly labeled conceptual schematics.

## Figure classes

### Class A: Benchmark evidence

Generated from stored benchmark tensors, manifests, and results. Must include dataset, split strategy, model, metric, seed, and code path.

Examples: benchmark tables, subject-level strip plots, ROC curves, confusion matrices, effect sizes with confidence intervals.

### Class B: Diagnostic refit

Recomputed from the same exported tensors or a controlled fixture to explain a result. Must say whether predictions were exported from the benchmark or recomputed by the diagnostic script.

Examples: ridge alpha sensitivity, residual spectra, permutation nulls, autocorrelation controls.

### Class C: Conceptual schematic

Explains architecture or intuition. Must be stamped `SCHEMATIC - NOT BENCHMARK EVIDENCE` or equivalent.

Examples: system block diagrams, signal flow diagrams, robot sense-plan-dose-control loops.

### Class D: Clinical or biological claim

Requires the highest labeling rigor: physiological units, colorbar ranges, limitation boxes, and explicit evidence boundaries.

Examples: EEG topomaps with real montage, time-frequency maps, clinical wound-healing effect summaries.

## Tool routing by figure type

Kahlus uses the standard neuroscience toolchain by data type. Do not use a domain-specific renderer unless the required data are present.

- **Benchmark/statistics/leaderboards:** use pandas + seaborn + matplotlib/tueplots. Store cached CSV/JSON in `data/`, source scripts in `src/`, and PNG/PDF/SVG in `figures/`, following the CEBRA figure-source pattern.
- **EEG/MEG/electrophysiology:** use MNE-Python conventions for real raw traces, epochs, ERPs, topomaps, PSD, time-frequency, or source estimates. Topomaps require real channel names and montage positions.
- **MOABB/Braindecode benchmark outputs:** use MOABB/Braindecode for data/model conventions, then pandas + seaborn for benchmark tables and statistical plots.
- **fMRI/connectome/anatomy:** use nilearn/nibabel for scriptable glass-brain, surface, statistical-map, and connectome figures. Use PyVista or FreeSurfer/PySurfer for real cortical meshes. Avoid screenshots from FSLeyes, ITK-SNAP, or 3D Slicer in reproducible `data→src→figures` pipelines.
- **NeuroML/simulation neuroscience:** use pyNeuroML/NeuroML, Brian2, or NEURON plotting for morphology, raster, voltage/current traces, F-I curves, and simulation protocols.
- **Architecture/conceptual schematics:** use draw.io, Inkscape, Illustrator, TikZ/PGF, PlotNeuralNet, or NN-SVG only for Class C schematics, clearly stamped as not benchmark evidence.
- **Colormaps:** use built-in perceptual colormaps such as `viridis` or `cividis` unless a domain-specific colormap is justified. Never use `jet`.

## Reputable visual patterns

### EEG / BCI / neurophysiology

Use MNE/MOABB/Braindecode conventions:

- raw traces are shown in real time units, with channel labels and scale clarity;
- topomaps require actual sensor names/positions, usually via a standard montage or dataset montage;
- PSD/time-frequency plots state frequency bands and sampling rate;
- model overlays include actual signal, prediction, residual, and a metric;
- benchmark plots distinguish train/val/test and avoid random-window leakage.

### NeuroML / computational neuroscience

Use NeuroML/pyNeuroML conventions:

- morphology figures come from cell/network model files, not decorative neuron clip art;
- voltage/current traces include units, stimulus timing, and protocol labels;
- F-I curves and electrophysiology summaries show the simulation protocol;
- diagrams should map directly to model components, LEMS/NeuroML files, or documented equations.

### Scientific Python documentation

Use Sphinx/MyST/PyData conventions:

- tutorials for first-run users;
- how-to guides for tasks;
- explanation pages for scientific reasoning;
- API/reference pages for code;
- citations and exact dependency versions for reproducibility.

## Slop detector

A figure is suspicious if it has any of these:

- no data source or provenance;
- no units on axes;
- synthetic-looking EEG waves with no sampling rate or channel identity;
- topomaps without montage/channel positions;
- unverified claims such as "ridge learns neural state" when it may be exploiting short-horizon autocorrelation;
- too many overlapping traces for readability;
- decorative gradients, glassmorphism, fake 3D brains, or icons replacing evidence;
- captions that explain vibes rather than methods.

## Kahlus figure rule

Every figure must be one of:

1. **Benchmark-derived evidence**: generated from stored tensors/manifests/results; includes source, split, model, and metrics.
2. **Diagnostic refit**: recomputed from the same exported tensors with code matching the benchmark baseline; clearly labeled.
3. **Schematic**: conceptual only; stamped `SCHEMATIC - NOT BENCHMARK EVIDENCE`.
4. **Clinical/biological claim**: supported by cited evidence and marked with limitations.

Anything else does not go into mentor-facing docs.
