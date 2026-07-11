# Kahlus HNPH working preprint

This directory contains a six-page working protocol preprint. It explains the
current repository state, the HNPH estimand, the Phase 0 study, and the work
required before any model or clinical claim.

Build from this directory with MacTeX or TeX Live:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error kahlus_hnph_preprint.tex
```

The checked-in PDF was compiled with the official NeurIPS 2025 style in
`preprint` mode. This is not a NeurIPS submission and contains no qualifying
empirical HNPH result.
