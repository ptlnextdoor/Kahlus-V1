# Claims

Allowed claim:

NeuroTwin proposes and evaluates a neural translation benchmark and model for heterogeneous neural signals under strict leakage controls.

Do not claim:

- first brain foundation model
- first multimodal brain model
- first stimulus-to-brain model
- clinical digital twin
- diagnostic model
- treatment predictor
- human brain clone
- proven neurology breakthrough

Synthetic smoke tests validate plumbing only. MOABB smoke tests validate the first real-data path only; they are not full NeuroTwin validation.

`tribe_style` is a NeuroTwin-native clean-room approximation for the TRIBE v2 stimulus-to-fMRI baseline lane. Do not describe it as exact TRIBE v2 or as using TRIBE v2 weights.

Scientific claims require real prepared data, leakage audit pass, validation-selected checkpoints, final held-out test metrics, and non-smoke run labeling through `scientific_claim_allowed=true`.
