# Kahlus Implementation State Ledger

Last updated: 2026-06-23
Workspace: `/Users/aayu/.codex/worktrees/8b64/Kahlus-V1`

This is the current implementation ledger for the active worktree: what exists, what is still left,
and what must stay blocked. It is a repo/worktree status document, not a claim of scientific
superiority or clinical readiness.

## Current Verdict

Kahlus is locally runnable and heavily instrumented, but it is not ready for A100 science runs or
clinical claims.

- v1 EEG future forecasting is the root lane and is currently the best-hardened path.
- The v1 smoke result is scientifically honest: `linear_ridge` still beats NeuroTwin/Kahlus.
- ResearchDock and NeuroVisual are useful side lanes, but neither should displace v1 EEG hardening.
- A100 packaging/audit code exists, but no A100 cluster job should launch from this dirty worktree.

Latest v1 EEG smoke metrics:

| metric | value |
| --- | --- |
| best baseline | `linear_ridge` |
| best baseline MSE | `0.4755090613404005` |
| TinySSM MSE | `0.8399924812887744` |
| shuffled-target MSE | `2.2021805365104257` |
| Kahlus beats best baseline | `false` |
| persistence/ridge dominate | `true` |
| shuffled target close to real baselines | `false` |
| scientific claim allowed | `true` |
| model-win claim allowed | `false` |
| claim scope | `eeg_future_forecasting_benchmark_ready` |

## Implemented

### v1 EEG Future-Forecasting Baseline Lane

Implemented:

- deterministic synthetic EEG fixture
- future-window forecasting task builder
- subject-held-out split audit
- HBN-style local manifest loader
- local manifest validation for paths, JSONL shape, IDs, channels, sampling rate, signal shape, and finite values
- baseline-first ladder including persistence, ridge, autoregressive ridge, TinySSM, TCN, Transformer, MLP, and NeuroTwin
- `shuffled_target_control` negative control
- train-only shuffled-target provenance
- autocorrelation diagnostics for short horizon, longer horizon, non-overlap, delta prediction, persistence/ridge dominance, and shuffled-target degradation
- target-scale context for normalized MSE interpretation
- per-subject, per-channel, and per-horizon metric sidecars
- `dataset_summary.json`
- `run_config.json`
- `baseline_table.json` and `baseline_table.csv`
- `baseline_verification.json`
- `baseline_checksum_manifest.json`
- local checksum audit script
- diagnostic report with artifact index, checksum audit, method order, run config, dataset summary, target-scale context, baseline gaps, model-win status, metric breakdown, gate criteria, split/gate/baseline failures, claim boundaries, autocorrelation diagnostics, and HBN local boundary when relevant
- semantic cross-artifact audits for metrics, reports, gates, failure reasons, dataset summary counts, target-scale context, autocorrelation rows, and manifest entries
- narrow evidence gate for `eeg_future_forecasting_benchmark_ready`
- model-win claim blocker when Kahlus does not beat the best baseline or autocorrelation baselines dominate

Primary files:

- `src/neurotwin/eeg_v1/`
- `scripts/run_eeg_v1_baselines.py`
- `scripts/run_eeg_autocorr_diagnostics.py`
- `scripts/audit_eeg_v1_baseline_checksums.py`
- `tests/eeg_v1/test_eeg_v1_sprint_a.py`
- `docs/roadmap/kahlus_v1_eeg_baseline_plan.md`

Current verification:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/eeg_v1 -v
```

### v1 EEG Few-Shot Adaptation Lane

Implemented:

- synthetic held-out-subject support/query adaptation task
- baselines first: `support_persistence`, `support_ridge`
- adapter candidates: `linear_probe`, `bottleneck_adapter`, `full_finetune`
- adaptation split audit
- adaptation dataset summary
- adaptation run config
- adaptation report with dataset, split, method order, ranking, gate criteria, and failure summaries
- adaptation checksum manifest and audit script
- narrow claim scope `eeg_fewshot_adaptation_benchmark_ready`

Primary files:

- `src/neurotwin/eeg_v1/adaptation.py`
- `scripts/run_eeg_v1_adaptation.py`
- `scripts/audit_eeg_v1_adaptation_checksums.py`
- `tests/eeg_v1/test_eeg_v1_sprint_b_adaptation.py`
- `docs/roadmap/kahlus_v1_fewshot_adaptation_plan.md`

Current verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

### ResearchDock Synthetic / Observation Lane

Implemented:

- deterministic synthetic ResearchDock sessions
- no-PII session schema
- reward/stress/social task templates
- synthetic data card
- response-profile metrics
- branch-specific evidence gate
- RD-1 protocol export and CSV export
- quality flags for missing/invalid pupil, invalid reaction time, invalid accuracy
- RD-2 subject-held-out observation task
- baselines before observation operator
- NumPy low-rank residual observation operator
- RD-3 public dataset mapping review without downloads/loaders
- RD-4 pilot preflight manifest and gate
- RD-5 profile-readiness audit without clustering
- RD-6 missing-modality audit
- RD-7 through RD-21 report/audit hardening: missing-modality counts, split summaries, split-audit sidecar, artifact indexes, gate criteria, failure reasons, data-card summary, session-grained quality counts, baseline ladder summary, and aggregate split-audit summary

Primary files:

- `src/neurotwin/researchdock/`
- `scripts/run_researchdock_synthetic.py`
- `tests/researchdock/`
- `docs/research/kahlus_affect_researchdock_roadmap.md`
- `docs/roadmap/kahlus_biomedical_execution_plan.md`

Current verification:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/researchdock -v
```

### NeuroVisual / NV-1 Metadata Lane

Implemented:

- metadata-only dataset registry for HBN-EEG, CHB-MIT, and TUSZ anchors
- verified/unverified/rejected dataset status fields
- no-download and no-A100 boundary flags
- metadata query plan artifact
- local manifest schema artifact
- local manifest validator CLI
- local split audit plan and validator CLI
- synthetic split manifest fixture
- registry evidence manifest with checksums
- registry bundle audit
- fixture replay
- handoff manifest builder and audit
- local evidence gate and evidence bundle package
- requirement coverage audit
- safe neurovisual ontology/condition mapping/intake tests

Primary files:

- `src/neurotwin/neurovisual/`
- `scripts/build_neurovisual_dataset_registry.py`
- `scripts/audit_neurovisual_local_manifest.py`
- `scripts/audit_neurovisual_local_split.py`
- `scripts/run_neurovisual_fixture_replay.py`
- `scripts/build_neurovisual_handoff_manifest.py`
- `scripts/audit_neurovisual_handoff_manifest.py`
- `scripts/run_neurovisual_local_evidence_gate.py`
- `scripts/package_neurovisual_local_evidence_bundle.py`
- `scripts/audit_neurovisual_requirement_coverage.py`
- `tests/neurovisual/`
- `docs/research/kahlus_neurovisual_dataset_registry.md`
- `docs/research/kahlus_neurovisual_epilepsy_roadmap.md`

Current verification:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/neurovisual -v
```

### A100 Handoff / Returned-Evidence Audit Preparation

Implemented locally:

- 7xA100 handoff package builder
- clean-worktree proof requirement
- runner tarball/checksum manifest contract
- CPU smoke command
- DDP/torchrun command
- no-secrets/no-checkpoints/no-raw-private-data checks
- returned-evidence audit with default expected GPU count of seven
- artifact package tests for handoff shape, dirty-worktree refusal, symlink refusal, and evidence audit behavior

Primary files:

- `src/neurotwin/a100_handoff.py`
- `src/neurotwin/a100_audit/auditor.py`
- `scripts/package_kahlus_a100_7x_handoff.py`
- `scripts/smoke_a100_runner.py`
- `scripts/audit_ktm_a100_evidence.py`
- `scripts/package_a100_evidence_bundle.py`
- `tests/artifacts/test_kahlus_a100_7x_handoff.py`
- `tests/artifacts/test_audit_ktm_a100_evidence.py`

Current verification:

```bash
PYTHONPATH=src python3 -m unittest tests.artifacts.test_kahlus_a100_7x_handoff tests.artifacts.test_audit_ktm_a100_evidence -v
```

### v2/v3/EM Synthetic Falsification Program

Implemented or present from earlier work:

- unified branch-aware evidence gate in `src/neurotwin/gates/`
- shared falsification core in `src/neurotwin/falsification.py`
- v2 dual-field synthetic model and falsifier
- v3 Transition Gym and KTM scaffolds
- KTM training harness under `src/neurotwin/training_v3/`
- EM Stage 0 no-human artifact audit/passive logging scaffolds
- synthetic-only smoke scripts and configs

Canonical details remain in:

- `docs/roadmap/kahlus_implementation_status.md`
- `docs/roadmap/sprint_ledger.md`

## Left To Do

### Immediate v1 EEG Work

- Keep hardening the v1 EEG evidence bundle before expanding architecture.
- Run local subject-held-out and baseline ladders on real user-provided local manifests when available.
- Add harder controls only when they have clear audit evidence, not just another report line.
- Keep ridge/persistence/TinySSM/shuffled-target controls first-class.
- Do not claim model win until Kahlus beats the best baseline under the evidence gate.

### v1 Adaptation Work

- Run the adaptation lane beyond synthetic fixture only after local manifest correctness is proven.
- Keep support-ridge/support-persistence as results, not scaffolding.
- Add adapter/model complexity only if the baseline ladder leaves a real gap.

### ResearchDock Work

- Apply the RewardDock clinical-extension docs patch when explicitly requested.
- Keep ResearchDock clinical/research-first and wellness-later.
- Do not implement hardware code yet.
- Do not frame cortisol/alpha-amylase as the product; they are optional future ingredients.
- Keep diagnosis/treatment/medication claims blocked.

### NeuroVisual Work

- Do not start NV-2 adapters until the root v1 EEG lane is solid.
- Future NV adapter work must start from verified local manifests, not invented paths or downloads.
- Baselines and split audits must precede models.
- No epilepsy diagnosis, symptom diagnosis, photic-trigger instructions, or clinical claims.

### A100 Work

A100 is not ready to launch from this worktree.

Required before any A100 job:

1. local synthetic fixture passes
2. subject-held-out split audit passes
3. baseline ladder runs locally
4. evidence gate passes
5. clean worktree
6. exact commit hash
7. handoff package rebuilt from clean merge/tag
8. CPU smoke test passes
9. 7xA100 labeling confirmed, never 8xA100
10. no secrets, checkpoints, raw private participant data, or raw public neural data in package
11. A100 smoke run returns evidence
12. returned evidence audit passes

### Repo / Process Work

- Decide what belongs in the next commit; the worktree is currently dirty with broad parent-thread changes.
- Stage explicitly by path; do not `git add -A`.
- Keep `graphify-out/` as a separate generated refresh if committed at all.
- Do not merge, open PR, or launch cluster work unless explicitly asked.

## Blocked Claims

Blocked:

- first brain foundation model
- first multimodal brain model
- first stimulus-to-brain model
- clinical digital twin
- diagnosis of depression, anhedonia, social anxiety, epilepsy, PTSD, or recovery
- treatment or medication recommendations
- model superiority/SOTA from current v1 EEG smoke
- A100-scale result claims from local-only runs
- cortisol/alpha-amylase alone measuring anhedonia

Allowed narrow wording:

- `eeg_future_forecasting_benchmark_ready`
- `eeg_fewshot_adaptation_benchmark_ready`
- synthetic-only ResearchDock/NeuroVisual/Transition Gym readiness language when the relevant gates pass
- local handoff/audit package readiness after clean local verification

## Current Verification Snapshot

Latest broad verification observed in this worktree:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Result: `505` tests passed, `2` skipped.

Latest v1 EEG verification observed:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/eeg_v1 -v
```

Result: `130` tests passed.

Latest smoke/audit evidence observed:

- `/tmp/kahlus_v1_a80_smoke`
- `/tmp/kahlus_v1_a80_autocorr`
- checksum audit passed with `17` artifacts checked
- `a100_jobs_launched=false`

Docs-only verification for this ledger:

```bash
git diff --check -- ':!graphify-out'
```

## Dirty Worktree Note

This worktree contains intentional modified/untracked sprint artifacts. This doc does not stage,
commit, revert, stash, merge, open a PR, or authorize A100 execution.
