# Upstream Registry

This repo can reuse upstream work aggressively, but license and access status must be explicit before code, weights, or data are copied.

| Asset | Role | Status |
| --- | --- | --- |
| TRIBE v2 | Stimulus/video/audio/language -> fMRI baseline | CC BY-NC reference; local `tribe_style` is clean-room approximation only |
| BrainVista | Naturalistic fMRI next-token baseline | Verify code/weights license before reuse |
| Brain-OF | fMRI/EEG/MEG foundation baseline, main boss fight | Verify code/weights license before reuse |
| BrainOmni | EEG/MEG tokenizer/model baseline | Verify code/weights license before reuse |
| Brain Harmony | Structure + function token baseline | Verify code/weights license before reuse |
| Mamba | SSM baseline/core candidate | Pin upstream commit before vendoring |
| NDT3 | Spike/population dynamics baseline | Pin upstream commit and license |
| Neuroformer | Spike transformer baseline | Pin upstream commit and license |
| CEBRA | Representation baseline | Apache-2.0 for recent versions; track patent notes |
| BrainLM | fMRI baseline/reference | Restricted CC BY-NC-ND; research-only adapter |
| SpikingJelly | SNN tooling | Custom license; quarantine adapter |

Restricted assets must live behind adapters and cannot become required for open benchmark reproduction.
