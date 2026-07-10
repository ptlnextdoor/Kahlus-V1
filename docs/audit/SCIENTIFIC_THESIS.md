# Scientific Thesis Reconstruction

## Narrowest Defensible Question

Does a learned sequence model capture predictive information in EEG that remains after persistence and linear autoregressive structure are controlled, when the target is strictly later than the observed context and evaluation holds out subjects, recordings, and datasets?

The current repository does not yet answer this question. Its recovered EEG task predicts a one-sample-shifted copy of the same window, so context and target overlap in 126 of 127 samples.

## Proposed Mathematical Object

For subject `s`, session `r`, montage `M`, and observed signal

`x_{s,r}: [0,T] x M -> R`,

the defensible object is a conditional predictor

`F_theta: (x|_[t-L,t], M, q) -> P(x|_[t+g,t+g+H] | x|_[t-L,t], M, q)`,

where `q` records signal quality and acquisition metadata, and `g > 0` prevents target overlap. A latent state `z_t` is scientifically useful only if it improves held-out prediction, missing-sensor reconstruction, or transfer beyond matched finite-dimensional baselines and nuisance controls.

## What the Current System Learns

The strongest recovered result used `NeuralStateSpaceTranslator` with a temporal-convolution encoder, a 12-layer GRU selected through the `ssm_fallback` name, and a linear observation head. It is a finite-dimensional sequence-to-sequence predictor. It is not evidence for Mamba, a state-space model in the modern structured-SSM sense, a continuous neural field, or a neural operator.

The separate experimental `NeuralFieldCompiler` constructs a latent tensor from one selected source modality, applies learned gated tensor dynamics and optional pair interactions, and renders outputs through modality heads. It is presently an experimental latent-state architecture tested primarily on synthetic fixtures.

## Meaning of "Field" and "Compiler"

The code currently uses "field" to mean a latent tensor indexed by time, target-like nodes, and latent features. It does not define a coordinate-conditioned continuous function over scalp, cortex, time, or frequency. Sensor coordinates are not inputs to the latent representation, and discretization invariance is not demonstrated.

"Compiler" is a project metaphor for translating partial observations through a shared latent representation into observable outputs. It is not a compiler in the formal programming-language sense. In a paper, the term must be defined operationally and marked as a proposed architecture name.

## Claim-Type Boundaries

- Prediction: implemented, but the central future-window interpretation is contradicted by target overlap.
- Reconstruction: synthetic and failed on the recovered real masked task.
- Cross-subject transfer: subject-separated machinery exists; no strong matched generalization result.
- Cross-dataset/site transfer: planned/infrastructure only.
- Symptom mapping and biomarker discovery: unsupported by clinical cohorts.
- Causal inference and perturbation response: synthetic/conceptual only.
- Clinical decision support: unsupported and blocked.

## Falsifiable Central Hypothesis

After enforcing `target_start >= context_end + gap`, fitting every transform on training subjects only, and evaluating at subject/recording level, an NFC-like model will improve forecast error or held-out-sensor likelihood over persistence, tuned ridge/AR/VAR, TCN, GRU, Transformer, and shuffled/time-shift controls across more than one dataset.

This hypothesis fails if gains disappear with non-overlapping targets, if subject/session nuisance baselines explain the result, if the latent state does not transfer across montages or datasets, or if a compute-matched conventional sequence model performs equally well.
