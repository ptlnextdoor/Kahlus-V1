# NeuroTwin Project State: NFC Pivot and Evidence-Gated Plan

## One-Paragraph Summary

NeuroTwin is a Python research package and CLI for leakage-audited neural translation experiments. The repo began as a benchmark scaffold for missing-modality reconstruction, future-state forecasting, cross-modal translation, few-shot subject adaptation, and dataset/site generalization under held-out splits. The current research direction keeps that infrastructure but makes NeuroTwin NFC, the Neural Field Compiler, the experimental model primitive: recordings are treated as partial, lossy observations of one evolving latent neural field rather than separate modalities to fuse directly. The repo is currently an evidence-gated research harness, not a completed model-superiority result.

## Architecture Hierarchy

| Layer | Role | Status |
| --- | --- | --- |
| NeuroTwin NFC / Neural Field Compiler | Core experimental architecture | implemented experimental path |
| NeuralStateSpaceTranslator | Legacy/current NeuroTwin baseline | retained baseline |
| NeuroTwinPairOperator | Historical pairwise operator | retained baseline and ablation |
| Ridge/autoregressive ridge/direct MLP/TCN/Transformer/SSM | Local sanity and sequence baselines | implemented baselines |
| BrainVista-style/TRIBE-style | Clean-room approximation lanes | approximations only |
| Manifest, leakage, claim gates, reports | Evidence infrastructure | trusted but must stay claim-hygienic |
| fNIRS/TurboVec/TurboQuant | Optional theory/infrastructure notes | docs only or deferred |

## Track A and Track B

Track A is the reproducibility and model-gates paper lane. It uses MOABB/EEG and synthetic smoke paths to validate leakage audits, identity probes, model cards, paper-mode reports, evidence bundles, and honest claim boundaries. Track A does not require NeuroTwin to beat ridge.

Track B is the model architecture paper lane. It uses NFC. The first proof point is strict synthetic latent-field recovery. Only after that gate passes should the repo move to Algonauts/CNeuroMod fMRI with verified real stimulus features, then later to 1x A100 debug and 6x A100 DDP.

## What Old NeuroTwin Is Now

`NeuralStateSpaceTranslator` remains the legacy/current NeuroTwin baseline. It is useful because it shares the task API, prepared-manifest path, and evaluation surface. It is not the future claim by itself.

## What Pair-Operator Is Now

`NeuroTwinPairOperator` is no longer the main primitive. It is retained as a historical baseline and as an ablation lane for low-rank relational updates inside the NFC question.

## What NFC Is Now

NFC is the experimental architecture path. It models a subject-specific latent field `F_s(x,t,omega)` and learns observation operators for signals such as fMRI, EEG, behavior, and stimulus-conditioned responses.

## What fNIRS Contributes

fNIRS supports the observation-operator worldview: physiology, optics, hemodynamics, and artifacts make measurement a lossy operator over latent activity. This repo does not claim fNIRS support, MDD support, clinical diagnosis, or private-data results.

## What TurboQuant/TurboVec Contributes

TurboQuant/TurboVec is optional future retrieval, compression, and semantic-audit infrastructure around NFC. It is not the core model contribution and must not become a required dependency.

## What Is Implemented

- `nt` CLI for doctor, data prepare/smoke/audit, split audit, estimate, train, eval, and report commands.
- Event manifests, split manifests, leakage audits, prepared-task generation, and claim gates.
- Synthetic and MOABB preparation lanes.
- Prepared-manifest training with checkpointing, resume, metrics, reports, and DDP hooks.
- Local baseline runners and competitor registry.
- NFC model modules under `src/neurotwin/models/nfc/`.
- NFC synthetic suite and falsification report.
- A100 packaging and handoff scripts.

## What Is Scaffolded

- BIDS/OpenNeuro derivative scanning without raw preprocessing.
- A100/H100 cluster templates.
- BrainVista/TRIBE-style approximation lanes.
- Paper-mode artifact contracts.

## What Is Theory Only

- Full fNIRS observation support.
- TurboVec/TurboQuant integration.
- Spike/calcium modality implementations.
- Exact upstream TRIBE v2, BrainVista, Brain-OF, BrainOmni, or Brain Harmony reproduction.

## What Is Archived/Deferred

- Pair-Operator as the main architecture.
- Clinical digital-twin language.
- Depression/MDD diagnostic claims.
- Raw public neural data in git.
- Full Algonauts/CNeuroMod model claims before strict gates.

## Next Experiment Ladder

1. Correct and trust the NFC synthetic falsification gate.
2. Run strict 1x A100 NFC synthetic diagnostic: `synthetic50`.
3. If needed and documented, escalate to `synthetic100`.
4. Only after pass, run Algonauts/CNeuroMod 1x debug with verified stimulus hashes.
5. Only after debug pass, consider 6x A100 DDP.
6. Draft model-paper claims only after held-out evidence and baselines pass.

## Claim Boundaries

Allowed: leakage-audited benchmark infrastructure, experimental NFC path, synthetic NFC smoke/falsification suite, MOABB Track A reproducibility evidence path, A100 handoff packaging from clean committed HEAD.

Not allowed: model superiority, proven NFC, NeurIPS-quality result, clinical diagnosis, fNIRS support, exact TRIBE v2 or BrainVista reproduction, first brain foundation model, or A100 success without evidence bundles.

## Kill Criteria

- NFC cannot beat direct baselines on true field-grounded synthetic tasks.
- NFC equals no-observation or no-pair ablations within the configured threshold.
- Metrics are NaN or missing.
- Shape mismatch is padded or broadcast instead of failing.
- Real stimulus hashes are unavailable for stimulus-to-fMRI claims.
- Held-out splits fail leakage audit.
- Evidence gates and final reports disagree.
