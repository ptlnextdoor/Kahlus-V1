# Claims

Allowed claim:

NeuroTwin proposes and evaluates a neural translation benchmark and model for heterogeneous neural signals under strict leakage controls. The current experimental architecture is NeuroTwin NFC, a Neural Field Compiler that treats recordings as partial observations of a latent neural field.

Additional allowed current claims:

- The repo implements leakage-audited neural translation benchmark infrastructure.
- The repo implements an experimental NFC path.
- The repo supports synthetic NFC smoke/falsification checks.
- The repo supports MOABB Track A leakage/reproducibility evidence.
- The repo can package A100 handoffs from clean committed HEAD.

Do not claim:

- NeuroTwin beats baselines
- NFC is proven
- NeurIPS-quality result
- first brain foundation model
- first multimodal brain model
- first stimulus-to-brain model
- clinical digital twin
- diagnostic model
- treatment predictor
- human brain clone
- proven neurology breakthrough
- depression/MDD classifier
- full fNIRS support
- exact TRIBE v2 reproduction
- exact BrainVista reproduction
- TurboQuant as the model contribution
- A100 success if no evidence bundle exists

Synthetic smoke tests validate plumbing only. MOABB smoke tests validate the first real-data path only; they are not full NeuroTwin validation.

NFC synthetic tests validate plumbing only. Pair-Operator is an ablation/baseline inside NFC, not the main architecture.

Old short-step NFC synthetic signals produced before the corrected falsification gate are invalidated as evidence. They may be referenced only as debug history.

Forecasting numbers produced under overlapping input/target windows (notably the
historical `6621642` `future_state_forecasting` sidecar MSE 3.116 / Pearson 0.972)
are **invalid as forecasting skill** as of 2026-07-21
(`docs/research/eval_leakage_audit_2026-07-21.md`). Do not cite them as model
performance.

`tribe_style` is a NeuroTwin-native toy clean-room approximation for the TRIBE v2 stimulus-to-fMRI baseline lane. Do not describe it as exact TRIBE v2, as using TRIBE v2 weights, or as using real video/audio/text encoders unless explicit pretrained stimulus features are loaded.

Real stimulus claims require a verified source artifact hash. `transcript_hash` and synthetic feature hashes are plumbing-only and do not make stimulus-to-fMRI claims eligible.

Scientific claims require real prepared data, leakage audit pass, validation-selected checkpoints, final held-out test metrics, and an explicit `scientific_claim_allowed=true` decision in `summary.json`. A passed colocated `paper_mode_gate.json` means the artifact contract is satisfied; reports must show that gate status separately and must not promote claim allowance from the gate alone.

Conditional future claims:

- NFC synthetic claim only after the strict synthetic gate passes.
- Algonauts/fMRI claim only after real stimulus hashes are verified and held-out splits pass.
- Model superiority only if NFC beats baselines and ablations under an evidence gate.
- Semantic leakage audit claim only after exact or validated retrieval audit runs.
