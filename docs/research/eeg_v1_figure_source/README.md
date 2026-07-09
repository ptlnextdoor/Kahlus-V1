# EEG v1 figure-source packet

This directory follows the CEBRA paper-figure pattern: cached data artifacts live in `data/`, figure source scripts live in `src/`, and rendered figures live in `figures/`.

## Files

- `data/task_results.csv`: normalized rows parsed from versions evidence `task_results.csv` artifacts.
- `data/baseline_ranking.csv`: normalized rows parsed from `baseline_ranking.csv` artifacts.
- `data/audits.csv`: leakage, eval, and paper-mode gate rows parsed from JSON artifacts.
- `data/inventory.json`: counts of evidence artifacts and whether raw tensor/prediction arrays exist.
- `data/provenance.json`: source root and renderer provenance.
- `src/_figure_style.py`: shared matplotlib/tueplots style and PNG/PDF/SVG save helper.
- `src/Figure1_eeg_v1_benchmark_overview.py`: matplotlib/seaborn/tueplots-style EEG metric trajectory plots.
- `src/Figure2_eeg_v1_audit_matrix.py`: compact audit-status and artifact-coverage plots.
- `src/Figure3_eeg_v1_baseline_ranking.py`: task-wise recovered-Kahlus-versus-baseline MSE bar plots.

## Regenerate

```bash
PYTHONPATH=src python scripts/render_eeg_v1_ridge_visuals.py   --versions-root /Users/aayu/Downloads/versions   --out-dir docs/research/eeg_v1_ridge_visuals
```

Public rule: no raw tensor or prediction-array artifact means no waveform overlay, no residual trace, and no clinical/physiology claim figure. Public evidence figures use standard matplotlib/seaborn/tueplots axes with constrained layout, CEBRA-style cached data, and built-in perceptual colormaps such as `viridis`/`cividis`, not hand-drawn box diagrams.
