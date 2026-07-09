# EEG v1 ridge waveform sanity diagrams

This packet contains explanatory diagrams for Amrith's requested sanity check: what goes into ridge regression, what it predicts, and why a one-step EEG benchmark can reward simple linear structure.

```{warning}
The saved versions evidence archive does **not** contain raw EEG windows or saved prediction arrays. These figures are generated from the in-repo benchmark code path `neurotwin.benchmarks.baseline_suite._make_paired_windows(seed=0)` and are therefore **benchmark-contract diagnostics**, not raw EEG evidence or clinical physiology figures.
```

## Files

- `src/_figure_style.py`: shared matplotlib/tueplots style and PNG/PDF/SVG save helper.
- `src/render_ridge_waveform_sanity.py`: matplotlib/seaborn/tueplots-style renderer.
- `data/ridge_waveform_sanity_summary.json`: exact source, shapes, and metrics for the rendered example.
- `figures/FigureS6_ridge_future_window_contract.{png,pdf,svg}`: shows `X = EEG[t0:t6, channels]` and `Y = EEG[t1:t7, channels]`.
- `figures/FigureS7_ridge_prediction_overlay.{png,pdf,svg}`: shows a one-channel target overlay, ridge prediction, persistence baseline, and residuals.

## Regenerate

```bash
PYTHONPATH=src python docs/research/eeg_v1_ridge_sanity_diagrams/src/render_ridge_waveform_sanity.py
```

## Interpretation

The diagrams show the main sanity-check issue: for future-state forecasting, the target is a one-step shift of the same smooth EEG-like window. That makes recent same-channel values highly informative. Ridge can perform well for a useful but narrow reason: it can exploit short-horizon autocorrelation and linear channel mixing. This does not by itself prove a richer neural-state model.
