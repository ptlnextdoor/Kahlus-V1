# Kahlus HNPH protocol preprint

This directory contains the canonical single-column HNPH v0.4 protocol/theory
preprint. It does not contain an empirical HNPH result.

Build from this directory with MacTeX or TeX Live:

```bash
make
```

`make` first runs the sole figure renderer against the frozen protocol and then
compiles the manuscript. The renderer writes opaque vector PDFs, 300-DPI PNGs,
`figure_manifest.json`, `FIGURE_PROVENANCE.md`, and the LaTeX captions consumed
by the paper. Final figure PDFs must not be edited by hand.

Figures live in `docs/figures/hnph_protocol/`, which is the single canonical
location; the manuscript reads them through `\graphicspath`. There is no second
copy under this directory.

To render against a locally qualified source packet, run:

```bash
python ../../../scripts/analysis/plot_hnph_preprint_figures.py \
  --protocol ../../../configs/protocol/hnph_phase0_v0.4.yaml \
  --data-root /path/to/local/qualified-dod \
  --out-dir ../../figures/hnph_protocol
```

The data root remains local and must never enter git. Without a qualified DOD
packet, the appendix uses a hash-bound, descriptive Sleep-EDF figure from the
July 2026 vector package. Its provenance explicitly marks it as a single-label
transport illustration, not repeated-rater construct evidence or an empirical
HNPH result. A qualified local DOD example replaces it automatically.

`FIGURE_SELECTION_REVIEW.md` and `figure_selection_review.json` record which
supplied figure candidates were retained or rejected and why. The checked-in
PDF uses the official NeurIPS 2025 style in `preprint` mode; this is not a
NeurIPS submission.
