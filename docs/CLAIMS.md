# Claims

Allowed claim:

NeuroTwin proposes and evaluates a neural translation benchmark and model for heterogeneous neural signals under strict leakage controls. The current experimental architecture is NeuroTwin NFC, a Neural Field Compiler that treats recordings as partial observations of a latent neural field.

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

NFC synthetic tests validate plumbing only. Pair-Operator is an ablation/baseline inside NFC, not the main architecture.

`tribe_style` is a NeuroTwin-native toy clean-room approximation for the TRIBE v2 stimulus-to-fMRI baseline lane. Do not describe it as exact TRIBE v2, as using TRIBE v2 weights, or as using real video/audio/text encoders unless explicit pretrained stimulus features are loaded.

Real stimulus claims require a verified source artifact hash. `transcript_hash` and synthetic feature hashes are plumbing-only and do not make stimulus-to-fMRI claims eligible.

Scientific claims require real prepared data, leakage audit pass, validation-selected checkpoints, final held-out test metrics, and an explicit `scientific_claim_allowed=true` decision in `summary.json`. A passed colocated `paper_mode_gate.json` means the artifact contract is satisfied; reports must show that gate status separately and must not promote claim allowance from the gate alone.
