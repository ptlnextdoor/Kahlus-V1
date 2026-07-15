# Ridge EEG headline figure

`fig_ridge_overlap_headline.{pdf,png}` — one 4-panel composite showing the
BNCI2014_001 ridge headline (r=0.87 at h=1) is an input/target overlap
artifact, not evidence of learned neural dynamics. See
`../INTERPRETATION.md` for the full writeup and
`fig_ridge_overlap_headline_caption.tex` for the drop-in LaTeX caption.

Regenerate with:

```bash
PYTHONPATH=src python3 scripts/analysis/build_bnci_ridge_tensors.py
PYTHONPATH=src python3 scripts/analysis/plot_ridge_paper_figure.py \
  --npz artifacts/ridge_bnci_real/ridge_bnci_tensors.npz \
  --summary artifacts/ridge_bnci_real/ridge_bnci_summary.json \
  --out artifacts/ridge_bnci_real/figures
```

Provenance: BNCI2014_001 (BCI Competition IV-2a) via local MOABB cache,
subject-held-out split (train 1-6, test 7-9), per-subject z-scoring, ridge
alpha=1e-2, window length 128 (512 ms) @ 250 Hz.
