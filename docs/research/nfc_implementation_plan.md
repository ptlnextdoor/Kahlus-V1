# NeuroTwin NFC Implementation Plan

Status: experimental architecture pivot plan

NeuroTwin NFC reframes the project around a Neural Field Compiler. Brain
recordings are treated as partial observations of one evolving latent neural
field, then compiled into fMRI, EEG, behavior, or stimulus-response predictions
through structured observation operators.

## What Stays

- Event manifests, split manifests, leakage audits, paper-mode gates, evidence
  gates, model cards, baseline runners, and A100 runner packaging remain the
  experiment scaffold.
- The current NeuroTwin translator remains available as `current_neurotwin` or
  `neurotwin` baseline infrastructure.
- TRIBE-style, BrainVista-style, Brain-OF-style, and BrainOmni-style lanes remain
  labeled approximations unless exact upstream code, weights, data, and protocols
  are explicitly integrated.

## What Becomes Baseline-Only

- Pair-Operator is no longer the main architecture. Pair-Operator is an
  ablation/baseline for the NFC low-rank relational kernel hypothesis.
- Direct prediction heads are baseline or ablation paths. NFC claims must route
  through latent field inference/update plus observation operators.
- Stimulus-to-fMRI alone is not novelty; TRIBE v2 and BrainVista make it a
  baseline lane.

## New NFC Modules

- `LatentNeuralField`: represents `F_s(x,t,omega)` with shape
  `[batch, time, nodes, latent_dim]`.
- `FieldUpdateOperator`: causal field evolution using current/past state,
  stimulus, anatomy, subject state, and optional low-rank pair kernel.
- `LowRankPairKernel`: low-rank relational update that can be disabled for
  ablation and must change node updates when enabled.
- Observation operators: fMRI, EEG, and behavior compilers from latent field to
  measured outputs.
- `StimulusConditioningOperator`: causal lag adapter; real stimulus claims require
  verified source artifact hashes.
- `UncertaintyMapHead`: region/time uncertainty and optional pair uncertainty.
- `NeuralFieldCompiler`: main experimental model with `model_id=neurotwin_nfc`
  and `model_status=experimental_architecture`.
- `synthetic_field`: synthetic latent-field generator for local proof before A100.
- `nfc_suite`: synthetic benchmark and falsification report.

## Code Not To Touch

- Do not delete old NeuroTwin, Pair-Operator, baseline, leakage, paper-mode,
  or runner-bundle code.
- Do not weaken claim gates, split audits, non-finite quarantine, model-card
  constraints, or raw-data rules.
- Do not implement real EEG-fMRI translation unless the data are synthetic.
- Do not commit raw public neural data.

## Local Validation Before A100

Run the synthetic NFC suite first:

```bash
PYTHONPATH=src python3 -m neurotwin.cli eval --suite nfc_synthetic --out-dir /tmp/neurotwin_nfc_synthetic --train-steps 1 --seed 0
```

Then run the normal local gates:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m neurotwin.cli doctor
bash scripts/run_smoke.sh /tmp/neurotwin_nfc_local_smoke
git diff --check
graphify update .
```

Future 1x A100 debug command, documented only:

```bash
PYTHON_BIN=python3 bash scripts/slurm/_train_a100_inner.sh configs/train/algonauts_field_compiler_debug.yaml "$RUN_ROOT" 1
```

## What Would Falsify NFC

- NFC cannot beat direct baselines on synthetic latent-field recovery.
- Full NFC equals no-pair NFC, making the pair kernel decorative.
- Full NFC equals observation-operator-free NFC, making compiler factorization
  decorative.
- Uncertainty does not correlate with error.
- Ridge or BrainVista-style baselines beat NFC on every real fMRI task.
- Real stimulus hashes are missing or unverified for stimulus-to-fMRI claims.
