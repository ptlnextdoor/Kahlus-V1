# Methods

NeuroTwin represents neural recordings as `NeuralEventBatch` objects and splits recording-level manifests before preprocessing or windowing. The experimental NFC path infers a latent neural field with shape `[batch, time, nodes, latent_dim]`, evolves it causally, and compiles it into modality-specific observations. Pair-Operator is retained as an ablation for low-rank relational field updates.

Prepared training uses `train` for optimization, `val` for periodic evaluation and best-checkpoint selection, and `test` only for final held-out reporting. Reports expose `selection_split`, `report_split`, `best_val_mse`, and final `test_*` metrics so checkpoint selection cannot silently use test data.

Real-data v1 artifacts are prepared manifests, not raw public data. MOABB and BIDS adapters write leakage reports, manifest hashes, and benchmark metadata into run artifacts.
