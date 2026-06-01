# Baselines

Local implemented baselines:

- Linear/ridge sanity baseline with finite-output checks and stable least-squares fallback.
- MLP baseline.
- TCN/Conv1D baseline.
- Transformer baseline.
- SSM fallback baseline.
- NeuroTwin translator.

Prepared benchmark coverage:

- Future-state forecasting: local baselines and NeuroTwin are ranked on identical prepared windows.
- Masked neural reconstruction: local baselines and NeuroTwin are ranked on identical prepared masks.
- Cross-modal translation: local baselines and NeuroTwin are ranked where paired modalities exist.
- Stimulus-to-fMRI response: `tribe_style` is a local toy clean-room baseline for task plumbing until real pretrained stimulus features are integrated.
- Supervised prepared-task artifacts include bootstrap confidence intervals for MSE and MAE.
- Few-shot subject adaptation: reported as an auxiliary support/query sanity metric, not a full adaptation claim.
- Adaptation reports fixed support sizes when enough held-out query windows exist; query/test labels are never used for adapter fitting.
- Dataset/site generalization: reported as an auxiliary source-to-target sanity metric, not a full external validation claim.
- Failed baselines are recorded in `baseline_failures.json` and excluded from rankings instead of contaminating metrics with NaN/Inf values.

Competitor references:

- TRIBE v2: stimulus-to-fMRI competitor. Exact reproduction status remains unavailable; local `tribe_style` is a toy clean-room approximation and does not use TRIBE code, weights, notebooks, configs, or heavy runtime dependencies.
- BrainVista: fMRI forecasting/stimulus-to-brain competitor. Exact reproduction status: unavailable until upstream code/data/protocol are integrated.
- Brain-OF: fMRI/EEG/MEG shared foundation competitor. Exact reproduction status: unavailable until upstream code/data/protocol are integrated.
- BrainOmni: EEG/MEG tokenizer/model competitor. Exact reproduction status: unavailable until upstream code/data/protocol are integrated.
- Brain Harmony: structure + function token competitor. Exact reproduction status: unavailable until upstream code/data/protocol are integrated.
- Braindecode and CEBRA: optional wrapper slots. Status remains unavailable unless dependencies and compatible protocols are installed.

Prepared reports distinguish `exact`, `local_baseline`, `clean_room_approximation`, `approximation`, and `unavailable`. Approximation rows are lanes for fair comparison, not claims of exact reproduction.

Do not claim full competitor reproduction unless exact code, data, and protocol are actually used.
