# Kahlus paper reference set for neuroML figure design

Date: 2026-07-08

Purpose: collect leading papers whose figure language is closer to what Kahlus docs should imitate than the current figures. This set is legal/open-access-first. No Sci-Hub or closed PDF resolver was used.

## Zotero import status

All nine papers below were imported into Zotero with **imported PDF attachments**. Use the Zotero items tagged `Kahlus figure references` and `PDF attached`.

| Key | Paper | Venue/status | Zotero item with PDF |
|---|---|---|---|
| `eegpt_neurips2024` | EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals | NeurIPS 2024 | `XDD849A9` |
| `seeg_electrode_variability_neurips2024` | Neural decoding from stereotactic EEG: accounting for electrode variability across subjects | NeurIPS 2024 | `CS6NIJZ6` |
| `labram_iclr2024` | Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI | ICLR 2024 spotlight | `CEANPB8C` |
| `population_transformer_2024` | Population Transformer: Learning Population-level Representations of Neural Activity | arXiv 2024, neural population modeling | `GKWNQARC` |
| `hybrid_ssm_neural_decoding_2025` | Generalizable, real-time neural decoding with hybrid state-space models | arXiv 2025, real-time decoding | `DFXWD6TE` |
| `braingpt_2024` | BrainGPT: Unleashing the Potential of EEG Generalist Foundation Model by Autoregressive Pre-training | arXiv 2024, EEG foundation model | `IWAIZGZR` |
| `reve_eeg_foundation_2025` | REVE: A Foundation Model for EEG, Adapting to Any Setup with Large-Scale Pretraining on 25,000 Subjects | arXiv 2025, EEG foundation model | `WWUNPDBH` |
| `csbrain_2025` | CSBrain: A Cross-scale Spatiotemporal Brain Foundation Model for EEG Decoding | arXiv 2025, EEG foundation model | `33RCE6ZF` |
| `cebra_2022` | Learnable latent embeddings for joint behavioral and neural analysis | Nature 2023 / arXiv 2022 | `T5I3R7FH` |

A backup copy of the downloaded PDFs is outside the git repo at:

```text
/Users/aayu/Downloads/kahlus_neuroml_figure_reference_pdfs_2026-07
```

## What their figures look like

### 1. They use one conceptual architecture figure, not five vague diagrams

EEGPT, LaBraM, Population Transformer, and hybrid SSM all open with a compact architecture schematic that names real tensors and operations:

- EEGPT shows linear probing, electrode locations, and scaling law curves.
- LaBraM shows EEG channel patching as the central operation.
- Population Transformer shows electrode positions plus temporal embeddings feeding a transformer.
- Hybrid SSM shows tokenization, cross-attention, and an SSM backbone with explicit timing.

Kahlus implication: our Figure 1 should be a single clean pipeline schematic with real nouns: `raw EEG windows`, `split manifest`, `ridge baseline`, `NeuroTwin model`, `leakage audit`, `held-out subject/site`. No generic “AI module” boxes.

### 2. Their result figures are multi-panel, axis-heavy, and variance-aware

Strong papers use small multiples, confidence bands, standard error bars, per-task panels, and ablations. The Population Transformer paper’s captions explicitly call out standard error across subjects and sample-efficiency bands. The NeurIPS sEEG paper uses mean ± SEM model comparisons. LaBraM/BrainGPT/EEGPT use pretraining-loss curves and downstream task scaling plots.

Kahlus implication: replace cartoon-like figures with benchmark panels:

- x-axis: dataset/task/horizon/seed/model size
- y-axis: MSE, Pearson r, AUC, accuracy, or calibration error
- points/bands: subjects, seeds, SEM/CI
- caption: exact split, N, seed count, artifact path

### 3. They separate anatomy/electrode geometry from model performance

EEGPT has an electrode-location figure. The sEEG NeurIPS paper is explicitly about electrode variability across subjects. CSBrain uses activation patterns by task/brain region.

Kahlus implication: if we show electrodes or brain regions, we need a real montage, coordinate table, channel labels, and units. Otherwise keep it schematic and label it as schematic.

### 4. Their captions explain the conclusion, not just the pixels

Good captions say what changed and why it matters: sample efficiency, scaling trend, electrode variability, data volume, pretraining size, or per-task activation differences.

Kahlus implication: captions should read like:

> `[Benchmark evidence] Linear ridge has median MSE X across N saved MOABB evidence bundles, but the versions archive lacks saved prediction arrays, so trace overlays remain schematic.`

Not:

> `Figure shows model predictions.`

### 5. They make unsupported claims hard to accidentally infer

The best papers ground claims in protocol: train/test splits, subject counts, benchmark tasks, and ablations. The CEBRA paper makes latent embeddings credible by tying them to decoding, consistency, topology, and controls. The sEEG paper frames electrode variability as the problem, not as a footnote.

Kahlus implication: each figure must have an evidence class label:

- `[Benchmark evidence]` for parsed real results.
- `[Diagnostic]` for leakage/autocorr/residual checks.
- `[Schematic]` for explanatory diagrams.
- `[Clinical/biological claim]` only with real physiological units and montage provenance.

## Immediate redesign target for Kahlus figures

The current F1-F4 evidence plots are scientifically honest but visually weak. The old S1-S4 schematics are closer, but still need top-paper polish. The next redesign should create:

1. **One NeurIPS-style pipeline schematic** with the actual Kahlus data/model/evaluation contract.
2. **One benchmark strip/scatter panel** from real saved task results with dots, medians, and CIs.
3. **One leakage/autocorrelation diagnostic panel** with the visual polish of S3 but backed by exported tensors.
4. **One residual/PSD/prediction diagnostic panel** like S5, but only after future runs save `x_test`, `y_test`, and `y_pred` arrays.
5. **One electrode/montage figure** only when channel names and positions are real.

## GitHub figure-code audit

I also scanned the papers and project pages for official repositories, cloned the inspectable repos, and checked whether they include actual paper-figure generation code.

Read the audit here: [GitHub figure-code audit](github_figure_code_audit.md).

```{toctree}
:hidden:
:maxdepth: 1

GitHub figure-code audit <github_figure_code_audit>
```

The headline finding is that most EEG foundation-model repos ship training/eval code plus static README images, not exact paper panel scripts. The exception worth copying is CEBRA: it uses a separate `cebra-figures` repo with cached HDF5/CSV result artifacts, Jupytext figure scripts, rendered notebooks, and CI.

## Files in this reference set

- `paper_manifest.json`: source URLs and download status.
- `zotero_pdf_attachment_verification.json`: Zotero item keys and PDF child keys.
- `figure_caption_extracts.json`: extracted figure-caption snippets for quick style review.
- `github_url_extracts.json`: GitHub/code URLs extracted from the downloaded PDFs.
- `github_figure_code_scan.json`: local scan results for plotting terms across cloned paper repos.
- `github_figure_code_audit.md`: human-readable audit of which repos actually show how figures are made.
- `MISSING_PDFS.md`: unresolved/missing PDF tracker.
