# Reputable neuroscience repo patterns

This page summarizes what high-trust neuroscience and scientific Python repositories look like, and what Kahlus should copy.

## EEG / BCI / neural decoding

### MNE-Python

- Repo: `mne-tools/mne-python`
- Role: canonical EEG/MEG/ECoG analysis and visualization toolkit.
- Docs pattern: Sphinx documentation, extensive examples, API reference, publication-ready figure tutorial.
- Figure lesson: use real data containers, channel metadata, sensor locations, time/frequency units, and reproducible plotting code.

### MOABB

- Repo: `NeuroTechX/moabb`
- Role: reproducible EEG/BCI benchmarking.
- Docs pattern: benchmark results, dataset summaries, examples, API, citation instructions.
- Figure lesson: benchmark visualizations must be tied to explicit datasets, paradigms, splits, and evaluation strategies.

### Braindecode

- Repo: `braindecode/braindecode`
- Role: PyTorch decoding models for raw EEG/ECoG/MEG with MNE/MOABB integration.
- Docs pattern: landing page, quickstart, tutorial gallery, model zoo, API, citation.
- Figure lesson: deep EEG figures should distinguish dataset loading, preprocessing/windowing, architecture, and evaluation.

### CEBRA

- Repo: `AdaptiveMotorControlLab/cebra`
- Role: latent embeddings for joint behavioral and neural analysis.
- Docs pattern: Sphinx/PyData theme, installation, usage, demos, figures, API docs, citations.
- Figure lesson: a serious method repo gives readers a path from paper figures to executable demos and APIs.

## NeuroML / computational neuroscience

### NeuroML documentation and pyNeuroML

- Repos: `NeuroML/Documentation`, `NeuroML/pyNeuroML`, `NeuroML/libNeuroML`
- Role: standard model descriptions, simulation, visualization, and analysis of neural models.
- Docs pattern: Sphinx/ReadTheDocs, specification pages, tutorials, software pages, provenance guidance.
- Figure lesson: morphology, voltage traces, F-I curves, and network diagrams should be generated from model files and simulation protocols.

## fMRI / neuroimaging

### Nilearn

- Role: fMRI/statistical neuroimaging plotting and machine learning.
- Docs pattern: example gallery, plotting API, decoding/connectivity tutorials.
- Figure lesson: brain images need atlas/template/projection provenance, threshold rules, and clear statistical meaning.

## Data and reproducibility infrastructure

### SpikeInterface, AllenSDK, DANDI, BIDS/MNE-BIDS

- Role: acquisition/analysis pipelines, public datasets, metadata standards.
- Lesson: serious neuroscience code emphasizes data provenance, metadata, file standards, and reproducible examples over decorative visuals.

## What Kahlus should copy

- Sphinx/MyST docs with a clean landing page.
- Figure pages that include generated assets, captions, and provenance.
- API reference generated from code once public interfaces stabilize.
- A tutorial path: install → prepare data → run baseline → inspect ridge diagnostics.
- A methods path: data contracts → split manifests → leakage audit → baselines → claims.
- Citation and evidence pages.

## What Kahlus should avoid

- Fake EEG waveforms generated directly by image prompts.
- Generic brain clip art.
- Topomaps without sensor positions.
- Claims not tied to exact run artifacts.
- Docs that look like a pitch deck but cannot reproduce a figure.
