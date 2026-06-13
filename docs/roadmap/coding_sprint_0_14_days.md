# Coding Sprint 0–14 Days

First sprint of the v2/v3/EM expansion. Output is **infrastructure + synthetic falsification
scaffolding**, not scientific results. v1 is frozen and untouched.

## What was implemented

1. **Unified evidence gate** — `src/neurotwin/gates/unified_gate.py`. Branch-aware
   (`v1|v2|v3|em`) gate writing/reading the dossier JSON schema. `scientific_claim_allowed`
   is True only when all required checks pass **and** `claim_scope` is in a narrow
   synthetic/audit allowlist. Separate from the v1 gate (zero risk to v1).
2. **v2 dual-field scaffold** — `src/neurotwin/models/dual_field/` (config, fast_field,
   slow_field, coupling, observation_heads, dual_field_compiler, stability). Deterministic,
   seeded, spectrally stabilized, finite-checked. EEG-like readout from the fast field N;
   BOLD/fNIRS-like readout from the slow field H.
3. **v3 Transition Gym** — `src/neurotwin/transition_gym/` (synthetic_worlds,
   perturbation_library, observation_compilers, splits, metrics, data_cards). Known hidden
   operators, locked non-commutative perturbation battery, EEG-like + behavior outputs,
   subject adapters, train/val/test splits, and a held-out perturbation-composition split.
4. **v3 KTM scaffold** — `src/neurotwin/models/ktm/` (config, history_encoder, memory,
   response_profile `C_K`, perturbation_operators `T_a`, lie_generators `[Ta,Tb]`,
   neuro_experts, uncertainty, ktm). Forward-pass only; non-commutative operators.
5. **Shared baseline runner** — `src/neurotwin/baseline_runner.py`. Runs ridge,
   autoregressive_ridge, mlp, transformer, ssm_fallback (honest GRU placeholder), and nfc
   (importable but skipped honestly) on the v2 and v3 tasks. Emits `metrics.json`,
   `baseline_table.{csv,json}`, `evidence_gate.json`, `run_config.json`, seed, failure reasons.
6. **Kahlus-EM Stage 0** — `src/neurotwin/em/` (context schemas, PSD/channel artifact features,
   descriptive EM-response metrics, artifact audit, offline geomagnetic loader, room EMF logger,
   EM gate). Artifact model documented: `Y_EEG_measured = Y_EEG_brain + A_sensor(E_t) + ε_t`.
7. **Smoke scripts + configs** — see implementation status doc for the file list.

## What remains (future sprints, not now)

- Wiring NFC as a first-class baseline in the shared runner (currently skipped honestly).
- Calibration computation so a narrow synthetic claim could legitimately pass its gate.
- Richer Transition Gym worlds (more operators, longer horizons, behavior tasks) and KTM training.
- fNIRS optical observation operator (documentation/stub track only).
- TurboVec/retrieval infrastructure (offline audit only) — still deferred.
- Any real-data (MOABB/Algonauts) or A100 work — gated behind passing local synthetic evidence.

## Tests to run

```
PYTHONPATH=src python3 -m unittest discover -s tests -v
# Focused:
PYTHONPATH=src python3 -m unittest tests.gates.test_unified_gate
PYTHONPATH=src python3 -m unittest tests.models.test_dual_field
PYTHONPATH=src python3 -m unittest tests.transition_gym.test_transition_gym
PYTHONPATH=src python3 -m unittest tests.models.test_ktm
PYTHONPATH=src python3 -m unittest tests.test_baseline_runner
PYTHONPATH=src python3 -m unittest tests.em.test_em_stage0
```

Smoke scripts (write artifacts under the given out-dir):

```
PYTHONPATH=src python3 scripts/run_dual_field_synthetic.py --out-dir /tmp/kahlus_v2_smoke --config configs/models/dual_field_synthetic.yaml
PYTHONPATH=src python3 scripts/run_transition_gym_baselines.py --out-dir /tmp/kahlus_v3_gym
PYTHONPATH=src python3 scripts/run_ktm_synthetic.py --out-dir /tmp/kahlus_v3_ktm
PYTHONPATH=src python3 scripts/run_em_artifact_audit.py --out-dir /tmp/kahlus_em_stage0 --config configs/em/stage0_artifact_audit.yaml
PYTHONPATH=src python3 scripts/run_em_passive_logging_analysis.py --out-dir /tmp/kahlus_em_stage1
```

## Claim boundaries

- Everything here is synthetic/scaffold/proposed. No model-success or SOTA claim.
- Every emitted evidence gate returns `scientific_claim_allowed=false` (no calibration computed),
  which is the correct, honest outcome for this sprint.
- No clinical, diagnostic, treatment, brain-control, consciousness, or God-Helmet claims.
- No `do(a)` / causal language without a real randomized/assigned intervention.

## No-A100 policy

No A100/cluster jobs are launched or required by this sprint. A100 work is gated behind passing
the local synthetic tasks, baselines, unit tests, and evidence gates added here.
