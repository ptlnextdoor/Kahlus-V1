# Visual standards: reputable neuroscience vs AI slop

The visual bar for Kahlus is: **every research figure must be generated from data, code, and provenance**. Prompted SVGs are allowed only for explicitly labeled conceptual schematics.

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

Anything else does not go into mentor-facing docs.
