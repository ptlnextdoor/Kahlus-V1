# Neural Field Compiler Validation

## Verdict

The name is a proposed research program, not a validated mathematical description of the current implementation.

## Required Definition

Let `D` be a physical domain containing sensor coordinates `p`, time `t`, and optionally frequency `f`. A neural field should define a function such as

`z_s(p,t,f) = phi_theta(p,t,f; observations, subject/session metadata)`.

An observation operator `O_m` maps the latent field to modality `m` at that modality's sampling coordinates. A neural operator claim additionally requires a learned map between function spaces whose parameters can be evaluated across discretizations, together with resolution or montage transfer evidence.

The model must state support, boundary assumptions, interpolation, identifiability limits, and uncertainty. EEG scalp potentials are volume-conducted mixtures; a latent state is not automatically a cortical source or physiological state.

## Current Implementation

- `NeuralFieldCompiler.forward_task` projects every available source modality to the target sensor set and fuses them with learned, reported softmax weights.
- Task embeddings distinguish forecasting, reconstruction, readout, and translation computation paths.
- Optional measured coordinates are validated and encoded per output sensor; configured geometry cannot be silently omitted.
- `LatentNeuralField` maps scalar node values to finite latent features.
- `FieldUpdateOperator` uses an actual selectable causal GRU or causally masked Transformer followed by same-time sensor attention.
- Optional structural matrices influence same-time attention and invalid shapes fail closed.
- The EEG observation operator pools latent nodes and applies a linear readout.
- The fMRI temporal adapter uses left-only support and an explicit nonnegative delay.
- Prepared training scores predictive scale with a Gaussian negative log-likelihood when uncertainty is enabled.
- Subject identifiers are rejected by the NFC forward path.

This remains a coordinate-conditioned finite latent-tensor sequence model. It may become a useful architecture, but field/operator terminology has not yet earned empirical support because no montage, resolution, or discretization-transfer experiment has passed. Gaussian likelihood training also does not establish held-out calibration.

## Required Falsification Experiments

1. Coordinate removal: compare a coordinate-aware model against the same model with shuffled/removed electrode coordinates.
2. Montage transfer: train on one montage and test on unseen channel subsets/densities.
3. Discretization: evaluate identical weights across sampling grids and channel densities.
4. Interpolation: reconstruct held-out sensors with distance-stratified error.
5. High-frequency recovery: quantify spectral attenuation and phase error.
6. Operator ablation: compare against a parameter/compute-matched GRU, TCN, and Transformer.
7. SIREN baseline: coordinate-to-signal interpolation on the same observed coordinates.
8. FNO/graph-operator baseline only on a task with justified common domain/discretization.
9. Subject/site nuisance probes on latent states.
10. Held-out calibration and prediction-set coverage after probabilistic training.

## Naming Rule

Until coordinate and discretization tests pass, manuscripts should say "experimental Neural Field Compiler (NFC), a latent-tensor observation model" and should not call it a neural operator, cortical field estimator, biological state recovery system, or digital twin.
