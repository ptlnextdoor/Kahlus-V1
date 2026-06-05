# NFC Architecture Decision Log

## Decision

The main NeuroTwin architecture is now NFC: a Neural Field Compiler. The core
object is a latent neural field. Pair-Operator is retained as one ablation inside
this architecture, not as the project spine.

## Why Old NeuroTwin Was Insufficient

The old formulation treated windows as inputs to a predictor. That was useful
for leakage-proof benchmark plumbing, but it did not separate hidden neural
dynamics from modality-specific measurement physics.

## Why Pair-Operator Alone Was Insufficient

Pair-Operator tested whether low-rank region relationships improve fMRI-first
forecasting. That is a real ablation, but it is not the deeper primitive. NFC
keeps low-rank pair updates as one possible field interaction kernel.

## Why NFC Is The New Primitive

NFC models many sensors as partial views of one field. The model infers and
evolves `F_s(x,t,omega)`, then observation operators compile that field into
fMRI, EEG, behavior, or stimulus-response outputs.

## Mathematical Versus Speculative

- Mathematical: latent field shape, causal update, low-rank kernel, observation
  operators, and falsification criteria.
- Speculative: whether the factorization improves real held-out fMRI or EEG
  benchmarks.

## Proof Requirements

- Synthetic: recover latent fields and show full NFC is not equivalent to
  no-pair or no-observation-operator ablations.
- Algonauts/fMRI: improve only under verified stimulus artifacts, leakage audits,
  and strong baseline rankings.
- Before superiority claims: pass real held-out subject/site/dataset splits,
  exact or honestly labeled baseline comparisons, bootstrap intervals, and
  explicit `scientific_claim_allowed=true` in `summary.json`.
