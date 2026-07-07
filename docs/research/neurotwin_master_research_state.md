# NeuroTwin Master Research State

## Current Repository State

NeuroTwin is a leakage-audited neural translation research package. The repo contains working infrastructure for schemas, split manifests, leakage audits, prepared tasks, local baselines, prepared training, reports, model cards, evidence gates, and A100 packaging. The experimental model path is now NeuroTwin NFC, the Neural Field Compiler.

The repo does not yet contain a model-superiority result. Synthetic and MOABB smoke runs are plumbing checks unless a final evidence gate explicitly allows a claim.

## Track A: Reproducibility and Gates

Track A validates the evidence system. It uses synthetic and MOABB/EEG paths to prove that leakage audits, identity probes, model cards, paper-mode reports, claim gates, and artifact bundles behave honestly. Track A can succeed even if ridge or other classical baselines win.

## Track B: NFC Architecture

Track B is the model path. NFC must first pass the synthetic latent-field gate. The next real-data battlefield is Algonauts/CNeuroMod-style fMRI with verified stimulus hashes and held-out splits. Only after synthetic pass and 1x debug should 6x A100 DDP be considered.

## Why Pair-Operator Was Demoted

Pair-Operator captured a useful low-rank relational update, but it is not broad enough to be the primitive for fMRI, EEG, behavior, stimulus response, and future modalities. It remains valuable as a baseline and ablation that tests whether pairwise field coupling matters inside NFC.

## Why NFC Became the Primitive

Direct fusion treats modalities as separate streams. NFC treats them as partial observations of one latent neural field:

```{math}
F_s(x,t,\omega)\in\mathbb{R}^d
```

```{math}
Y_m=\mathcal{O}_m(F_s,A_s,U,\epsilon_m)
```

The bet is that field-mediated translation, `y_a -> F -> y_b`, generalizes better than direct translation, `y_a -> y_b`, when splits are leakage-safe.

## Equation Categories

The no-loss equation ledger is `docs/research/equation_ledger.md`. It covers:

- Transformer and Mamba baseline schematics.
- Latent field definition.
- Observation operators.
- Generative probabilistic model.
- Controlled dynamics and neural CDE-style updates.
- Neural-field PDE and discretized dynamics.
- Low-rank pair kernels.
- Graph derivative, Laplacian, and graph regularizers.
- HRF/fMRI, EEG, MEG, spike, calcium, behavior, and fNIRS operators.
- Variational posterior and ELBO.
- Gauge ambiguity and identifiability.
- Temporal, spectral, stability, and uncertainty regularizers.
- InfoNCE, Koopman residual dynamics, Riemannian EEG covariance, sampling lattices, Fourier features, coprime windows.
- TurboQuant/TurboVec math.
- Retrieval-kNN and semantic duplicate audit equations.
- Full NFC loss sketches and A100 experiment rules.

## fNIRS Research Status

fNIRS strengthens the observation-operator worldview because optical density is not brain activity directly; it is a physiology- and artifact-mediated measurement. The repo stores fNIRS as theory notes only. It does not claim fNIRS support, MDD support, diagnosis, or private-data results.

## TurboQuant/TurboVec Research Status

TurboQuant/TurboVec is optional future infrastructure for compression, retrieval baselines, latent memory, and semantic duplicate audits. It is not the core NeuroTwin model contribution. If implemented later, the first step should be a numpy exact vector store, then an optional lazy TurboVec adapter.

## Current Implementation Map

- NFC model: `src/neurotwin/models/nfc/`
- NFC synthetic suite: `src/neurotwin/benchmarks/nfc_suite.py`
- Synthetic field generator: `src/neurotwin/data/synthetic_field.py`
- Architecture registry: `src/neurotwin/models/architecture_registry.py`
- Old translator: `src/neurotwin/models/torch_models.py`
- Pair-Operator: `src/neurotwin/models/pair_operator.py`
- Leakage audits: `src/neurotwin/data/audit.py`, `src/neurotwin/eval/audit.py`
- Claim gates: `src/neurotwin/reports/evidence_gate.py`, `src/neurotwin/eval/paper_gate.py`
- A100 handoff: `scripts/a100_krish_agent_autorun.sh.in`, `scripts/package_a100_handoff_zip.sh`

## Immediate Next Step

Trust the synthetic falsification gate. The next A100 action is strict 1x NFC synthetic diagnostic, not Algonauts, not 6x DDP, and not a model-superiority run.

## Falsification Criteria

The NFC gate must return non-pass if:

- A baseline leaks `y_test`.
- A prediction shape mismatch is padded, broadcast, or silently transformed.
- A true latent-field task has no field target or explicit latent readout.
- Required metrics are missing.
- Metrics contain NaN or Inf.
- `nfc_full` does not beat direct baselines on the true field-grounded task.
- `nfc_full` does not show meaningful gain over no-observation or no-pair ablations.
- Uncertainty metrics are requested but non-finite.
- The ablation table is missing or invalid.

## A100 Run Order

1. Verify SHA256.
2. Extract handoff.
3. Run `./run_krish_agent.sh synthetic50`.
4. If documented, escalate to `synthetic100` only after `needs_evidence`.
5. If and only if synthetic passes, proceed to Algonauts/CNeuroMod 1x debug.
6. If debug passes, consider 6x A100 DDP.

Raw local equivalent:

```bash
export RUN_ROOT=/raid/scratch/$USER/neurotwin-nfc-synthetic-<short_sha>
PYTHONPATH=src python3 -m neurotwin.cli eval \
  --suite nfc_synthetic \
  --out-dir "$RUN_ROOT" \
  --train-steps 50 \
  --seeds 0 1 2 \
  --require-pass
```

## Claim Hygiene

Allowed current claims:

- The repo implements leakage-audited neural translation benchmark infrastructure.
- The repo implements an experimental NFC path.
- The repo supports synthetic NFC smoke/falsification checks.
- The repo supports MOABB Track A reproducibility evidence.
- The repo can package A100 handoffs from clean committed HEAD.

Not allowed:

- NeuroTwin beats baselines.
- NFC is proven.
- NeurIPS-quality result.
- Clinical diagnosis or treatment prediction.
- Depression/MDD classifier.
- Full fNIRS support.
- First brain foundation model.
- Exact TRIBE v2 or BrainVista reproduction.
- TurboQuant as the model contribution.
- A100 success without evidence artifacts.
