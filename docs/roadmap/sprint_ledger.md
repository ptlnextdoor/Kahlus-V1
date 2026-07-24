# Kahlus v2/v3/EM Sprint Ledger

Canonical, durable record of the v2/v3/EM synthetic-falsification program. Read this first when
picking up a fresh branch — it is the context capsule that survives branch deletion. Lives on
`main`, kept current as sprints land.

> **Locked wording.** A sprint "falsifier passed" means a *synthetic* benchmark survived honest
> tests on synthetic data. It does **NOT** mean Kahlus v2/v3 is validated on real brain data, nor
> any clinical/control/consciousness claim. "v2/v3 deserves life" = the synthetic branch earns
> further synthetic work, nothing more.

## Lane map

| Lane | Meaning | State |
|------|---------|-------|
| **v1** | built, evidence-gated EEG/fMRI stack (NFC, pair-operator, leakage splits) | **FROZEN** — never modified by v2/v3/EM work |
| **v2** | dual-field bridge (fast neural field N + slow hemodynamic field H) | proposed; **synthetic falsifier PASSED** (Sprint 1A) |
| **v3** | Transition Gym + KTM (hidden operators, perturbation algebra) | proposed; **synthetic operator-recovery falsifier PASSED** (Sprint 1B) |
| **EM** | Kahlus-EM no-human artifact audit (v3 side module) | Stage 0 scaffold only; report generator = Sprint 1C (next) |
| **v2.5** | Orch-OR / quantum-biology substrate motivation | **doc-only, parked** (Issue #15); gated behind v2 usefulness |

## Sprint trail (all merged to `main`)

| Sprint | PR | Tag | Sprint commit | Merge commit | Tests | Result |
|--------|----|-----|---------------|--------------|-------|--------|
| **0 — scaffolds** | #13 | `kahlus-sprint0-scaffolds` | `dbb05be` | `a0d2eaa` | 337 OK | infra only, all gates block claims |
| **1A — v2 falsifier** | #16 | `kahlus-sprint1a-v2-dual-field` | `70f246f` | `a538f29` | 348 OK | v2 falsifier **PASSED** |
| **1B — v3 falsifier** | #17 | `kahlus-sprint1b-v3-operator-recovery` | `c8f9917` | `4156778` | 361 OK | v3 falsifier **PASSED**; KTM loses to baselines (honest) |
| **1B.5 — falsification core** | #18 | `kahlus-sprint1b5-falsification-core` | `354f5a3` | `1eaca2f` | 369 OK | behavior-preserving dedup |

### Sprint 0 — v2/v3/EM synthetic scaffolds + unified evidence gate
Delivered the skeleton (infrastructure only, every gate blocks claims by default):
- `src/neurotwin/gates/` — branch-aware unified evidence gate (dossier JSON schema), separate
  from v1 `reports/evidence_gate.py`.
- `src/neurotwin/models/dual_field/` — v2 synthetic fast/slow dual-field system.
- `src/neurotwin/transition_gym/` + `src/neurotwin/models/ktm/` — v3 gym + KTM scaffold.
- `src/neurotwin/baseline_runner.py` — shared baseline sweep (reuses existing baselines/metrics).
- `src/neurotwin/em/` — Kahlus-EM Stage 0 no-human artifact audit + passive logging.
- `src/neurotwin/numerics.py` — suppress spurious Apple-Accelerate matmul FP warnings (values unchanged).
- scripts `run_*.py`, configs `configs/{models,em}/`, `docs/roadmap/`, deterministic seeded tests.
- Filed Issue #14 (Sprint 1 hardening) + Issue #15 (Orch-OR v2.5, doc-only).

### Sprint 1A — harden v2 dual-field into a falsifier
`src/neurotwin/models/dual_field/diagnostics.py` + `benchmark.py`. Seven diagnostics, all PASS
(seed-robust, adequate data budget): fast latent recovery (N), slow latent recovery (H),
EEG↦fast-field dependence, BOLD↦slow-field dependence, lag recovery (BOLD is delayed not
instantaneous), one-field vs two-field forecast (structure beats single-timescale without
regressing the slow channel), long-rollout stability. Gate allows narrow scope
`synthetic_dual_field_recovery`. Degenerate/tiny data correctly FAILS (no forced win).

### Sprint 1B — harden v3 Transition Gym operator recovery
`src/neurotwin/transition_gym/operator_recovery.py` + `benchmark.py`. Falsifier PASSED:
hidden-operator recovery (recover `M_k` from latent transitions vs known truth), held-out AB/BA
composition recovery (single-op estimates compose to unseen pairs), explicit non-commutativity
score, response-profile distances (trajectory / operator-induced separability / subject-transfer).
Baseline leaderboard incl. `retrieval_knn` + untrained KTM; **KTM loses to ridge** —
`ktm_beats_baselines=false`, reported honestly. This validates the **benchmark / operator-recovery
machinery, NOT v3 model superiority**. Gate scope `synthetic_transition_operator_recovery`.
No-cheat property verified: hidden operators used only for grading, never as model input.

### Sprint 1B.5 — extract shared falsification core (behavior-preserving)
Thermo-nuclear review of #17 found v2+v3 hand-copied the same harness. Extracted neutral
`src/neurotwin/falsification.py` (`Outcome`, `outcomes_finite`, `assemble_gate`, `build_report`,
`write_report`). v2+v3 migrated; duplicate Outcome types aliased; two divergent finite walkers
collapsed to one; canonical `regression_metrics` replaces a duplicate `_metrics`; `retrieval_knn`
registered in `baseline_runner`; task-config `Any` hints tightened. Same tests, same gate JSON,
same smoke verdicts. See `docs/research/falsification_core.md`.

### Sprint 2A — cluster-native trainable KTM harness (implemented; NOT yet merged)
The v3 "launchpad": a trainable KTM + standalone training harness so the gym task can move from
*untrained scaffold* to *trained model*, cluster-ready but run locally first. Landed code:
- `src/neurotwin/models/ktm/torch_ktm.py` — `TorchKTM` (trainable `nn.Module`) + `TorchKTMConfig`,
  a torch sibling of the numpy scaffold (scaffold left intact; still the operator-recovery grader).
- `src/neurotwin/training_v3/` — standalone harness (`config`, `dataset`, `objective`, `checkpoint`,
  `metrics_eval`, `trainer`, `bundle`). Isolated from the frozen v1 `training/` package; reuses
  `runtime/distributed.py` (DDP), `repro.py`, `falsification`/`gates`, and `baseline_runner.py`.
  Modes: `cpu_smoke` / `single_gpu` / `ddp`. Guards: finite/NaN micro-batch skip + loss-explosion
  abort. Checkpoint save/resume via `torch.save` / `torch.load(weights_only=True)` (+ fallback).
- `scripts/run_ktm_train.py`, `configs/train/ktm_synthetic_smoke.yaml`, `configs/train/ktm_a100_micro.yaml`.
- `gates/unified_gate.py` — added narrow scope `synthetic_ktm_training_harness` (additive).

**Two-tier claim discipline.** Primary scope `synthetic_ktm_training_harness` = infrastructure
readiness only (training runs, loss decreases, checkpoint/resume, bundle writes, DDP command exists,
gates block broad claims). The stronger `synthetic_ktm_recovery` stays **blocked** until a trained
KTM beats strong baselines on locked held-out metrics — a decreasing loss never flips it. CPU smoke:
val MSE 0.218 → 0.012, harness scope allowed, KTM honestly loses to the `mlp` baseline so recovery
stays blocked. Full suite 398 OK. No commit/PR/A100 in the implementing run (pending human review).

**A100 gate (unchanged, restated).** No cluster run until: synthetic task locked → baselines →
evidence gates → CPU smoke + ≥3 local seeds pass → complete output bundle (cards/metrics/failure
logs). The first cluster job is a tiny 8×A100 micro-sweep (short steps, 3 seeds, full bundle, no
claims beyond the gate), launched via the printed `torchrun --nproc_per_node=8` command (or the
slurm wrapper) — never a full beast run. 2A builds the launchpad; the rocket waits on the pad.

### Sprint 2D — KTM recovery-claim fairness fix + runbook tightening
The first real A100 micro-sweep (7×A100, commit `7915929`) was an **infra PASS** but wrongly flipped
`recovery_claim_allowed=true`: KTM test MSE `0.000696` beat the best baseline `mlp` `0.000947`, but
the baselines had run the runner's default **60** steps while the KTM ran **400** under 7-way DDP — a
training-budget artifact, not an earned recovery. Fix locks the comparison:
- `training_v3/config.py` — `baseline_train_steps` (0 = auto-match KTM `steps`) + `recovery_margin`
  (relative MSE the KTM must clear; default 0.05).
- `training_v3/metrics_eval.py` `ktm_vs_baselines` — records full budget provenance
  (`ktm_train_steps`, `ktm_world_size`, `ktm_global_batch_size`, `baseline_train_steps`,
  `baseline_batch_size`, `baseline_budget_policy="matched_optimizer_steps"`); `ktm_beats_baselines`
  is now an **earned** win = `comparison_locked AND relative_improvement >= margin`. World size is
  recorded but never inflates the baseline budget (would overfit the tiny synthetic baselines).
- `training_v3/bundle.py` — runs baselines to the matched budget; emits explicit recovery blockers
  (unmatched budget / sub-margin) so `failure_reasons.json` is auditable.
- `configs/train/ktm_a100_micro.yaml` (`baseline_train_steps: 400`, `recovery_margin: 0.05`) +
  smoke config (`recovery_margin: 0.05`).
- `deploy/a100/AGENT_RUNBOOK.md` — `python3` (host has no bare `python`), GPU-count honesty (label N×A100, never
  claim 8 when you ran 7), matched-budget + margin claim rule.
No A100 re-run from here (synthetic discipline); validated by CPU smoke + new `test_metrics_eval.py`
(the unmatched-budget regression stays blocked even when `ktm_mse < baseline_mse`).

## Roadmap ahead (not yet built)

- **Sprint 1C — EM Stage 0 report generator.** Prettier artifact report, artifact severity score,
  channel/frequency contamination map, HTML/Markdown report from phantom/no-human data. Reuse the
  falsification core (`src/neurotwin/falsification.py`) — do NOT invent a fourth harness. Still no
  humans, no stimulation.
- **Sprint 1D — baseline leaderboard + gate report polish** (per Issue #14).
- **v2.5 (Issue #15) — Orch-OR / quantum-biology appendix.** Doc-only, speculative substrate note
  (`Q_t` latent layer documented, never coded). Gated behind v2 proving useful.
- Real-data (MOABB/Algonauts) or A100 work: gated behind passing local synthetic falsifiers.

## Hard constraints (every sprint)

- **v1 is frozen.** Never modify load-bearing v1 paths: `training/prepared.py`,
  `benchmarks/suite.py`, `data/split_manifest.py`, `reports/evidence_gate.py`, eval command
  routing, `models/__init__.py`.
- No A100/cluster jobs until local synthetic falsifiers pass. No fake/placeholder "wins".
- No `do(a)`/causal wording unless intervention is actually randomized/assigned.
- EM = no-human only: no stimulation, high voltage, plasma/coils/gas, God Helmet, clinical claims.
  Geomagnetic fetch stays offline (no network).
- No new heavy deps (numpy/torch/PyYAML + stdlib only; no scikit-learn — cluster-only).
- Narrow synthetic claim scopes only; the gate blocks anything broader.

## PR / branch discipline

- One PR per sprint, single-purpose. Refactors get their own PR (do not fold into a feature PR).
- Pattern: build on a fresh branch off updated `main` → open PR → human review → merge commit
  (never squash; the sprint commits are research checkpoints) → tag the merge commit
  `kahlus-sprint<N>-<slug>`.
- Create branches with an explicit upstream (`git push -u origin <branch>`); branches made via
  `git checkout -b <b> origin/main` track `origin/main`, so a bare `git push` would target main.

## Commit granularity (Sprint 1C onward)

Sprints 0–1B.5 landed as one fat code commit each (e.g. Sprint 0 = 56 files in `dbb05be`). That
is hard to review and bisect. **From Sprint 1C on, split each sprint into small logical commits.**

- **One commit per logical unit** — a module/concern + its test together, not a sprint-wide blob.
  Target ~5–8 commits per sprint.
- **Conventional-commit prefixes**: `em:`, `v3:`, `v2:`, `gates:`, `tests:`, `docs:`, `chore:`.
  Example Sprint 1C: `em: artifact severity score` · `em: channel/frequency contamination map` ·
  `em: html/markdown report generator` · `em: wire report into stage 0 script` ·
  `tests: em report generator` · `docs: em report notes`.
- **Merge-commit, never squash** — so the small commits survive on `main`.
- The `graphify-out/` refresh is always its own separate `chore:` commit (never mixed with code).
- **Do not rewrite merged history** to re-split old fat commits — that breaks tags + merged-PR
  refs. Granularity applies forward only.

## Graphify discipline

- AGENTS.md: after editing code, run `graphify update .` (AST-only, no API cost) to refresh the
  knowledge graph in `graphify-out/`.
- `graphify-out/` is generated. **Exclude it from feature/refactor commits** (its trailing
  whitespace fails `git diff --check` and its churn bloats review). Stage source paths explicitly,
  never `git add -A`.
- When asked to "commit all changes", land the `graphify-out/` refresh as a separate
  `chore: refresh graphify knowledge graph after Sprint <N>` commit.

## Verification (run before every PR)

```
PYTHONPATH=src python3 -m unittest discover -s tests -v      # full suite, must stay green
PYTHONPATH=src python3 -m neurotwin.cli doctor               # v1 sanity unchanged
PYTHONPATH=src python3 scripts/run_dual_field_synthetic.py --out-dir /tmp/v2     # v2 falsifier
PYTHONPATH=src python3 scripts/run_transition_gym_baselines.py --out-dir /tmp/v3 # v3 falsifier
PYTHONPATH=src python3 scripts/run_ktm_synthetic.py --out-dir /tmp/ktm
PYTHONPATH=src python3 scripts/run_em_artifact_audit.py --out-dir /tmp/em0
PYTHONPATH=src python3 scripts/run_em_passive_logging_analysis.py --out-dir /tmp/em1
git diff --check -- ':!graphify-out'                         # source clean
```
Acceptance: full suite green; every written gate JSON has the exact dossier fields and
`scientific_claim_allowed=false` unless a narrow synthetic claim's required checks all pass; no v1
load-bearing file modified.
