# Baselines

Local implemented baselines:

- Linear/ridge sanity baseline.
- MLP baseline.
- TCN/Conv1D baseline.
- Transformer baseline.
- SSM fallback baseline.
- NeuroTwin translator.

Prepared benchmark coverage:

- Future-state forecasting: local baselines and NeuroTwin are ranked on identical prepared windows.
- Masked neural reconstruction: local baselines and NeuroTwin are ranked on identical prepared masks.
- Cross-modal translation: local baselines and NeuroTwin are ranked where paired modalities exist.
- Supervised prepared-task artifacts include bootstrap confidence intervals for MSE and MAE.
- Few-shot subject adaptation: reported as an auxiliary support/query sanity metric, not a full adaptation claim.
- Dataset/site generalization: reported as an auxiliary source-to-target sanity metric, not a full external validation claim.

Competitor references:

- TRIBE v2: stimulus-to-fMRI competitor.
- BrainVista: fMRI forecasting/stimulus-to-brain competitor.
- Brain-OF: fMRI/EEG/MEG shared foundation competitor.
- BrainOmni: EEG/MEG tokenizer/model competitor.
- Brain Harmony: structure + function token competitor.

Do not claim full competitor reproduction unless exact code, data, and protocol are actually used.
