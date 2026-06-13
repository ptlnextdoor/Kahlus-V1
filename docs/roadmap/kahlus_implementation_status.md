# Kahlus Implementation Status

Single source of truth for lane status. This sprint adds **infrastructure and synthetic
falsification scaffolding only** — no scientific victory, no A100 jobs.

## Lane status

| Lane | State | This sprint |
|------|-------|-------------|
| **v1** | **built / evidenced — FREEZE** | Untouched. Current evidence-gated EEG forecasting result lane is preserved. Do not mix v2/v3/EM into the v1 paper or result path. |
| **v2** | **proposed bridge — synthetic falsifier PASSED** | Sprint 0 scaffold + Sprint 1A falsification benchmark. On the synthetic system the dual-field split is recoverable and meaningful (fast/slow latent recovery, EEG↦N / BOLD↦H dependence, lagged neural→hemo path, two-field beats one-field, stable long rollout). Narrow synthetic claim (`synthetic_dual_field_recovery`) passes its gate with adequate data; degenerate/tiny data correctly fails. Still synthetic only, no real data. |
| **v3** | **proposed moonshot — synthetic operator-recovery falsifier PASSED** | Sprint 0 Transition Gym + KTM scaffold, hardened in Sprint 1B. On the synthetic gym the hidden operators are recoverable, held-out AB/BA compositions are predicted, the battery is non-commutative, and response profiles are separable. Baseline leaderboard incl. retrieval_knn + KTM; the **untrained KTM scaffold loses to ridge** (reported honestly) — this validates the benchmark/operator-recovery machinery, **not** v3 model superiority. Narrow scope `synthetic_transition_operator_recovery` passes its gate. Still synthetic only. |
| **EM** | **v3 side module — artifact audit first** | Stage 0 no-human artifact-audit + passive-logging scaffolds only. Report generator (severity score, contamination map, HTML/MD) is Sprint 1C (not yet built). |

Sprint 1B.5 extracted a shared, lane-neutral falsification core (`src/neurotwin/falsification.py`)
that v2 and v3 benchmarks both use; future lanes (EM Sprint 1C) reuse it. See
`docs/roadmap/sprint_ledger.md` for the full trail (tags, PRs, verdicts) and
`docs/research/falsification_core.md` for the core API.

```
v1 = built/evidenced, freeze current result lane
v2 = proposed bridge, synthetic only
v3 = proposed moonshot, synthetic Transition Gym first
EM = v3 side module, artifact audit first
```

## What exists after this sprint

- Unified, branch-aware evidence gate: `src/neurotwin/gates/` (dossier schema:
  `branch, dataset, split_audit_passed, baseline_table_present, finite_metrics,
  calibration_checked, claim_scope, scientific_claim_allowed, failure_reasons`). This is
  **separate** from the load-bearing v1 gate at `src/neurotwin/reports/evidence_gate.py`.
- v2 dual-field synthetic scaffold: `src/neurotwin/models/dual_field/`.
- v3 Transition Gym: `src/neurotwin/transition_gym/`; KTM scaffold: `src/neurotwin/models/ktm/`.
- Shared baseline runner: `src/neurotwin/baseline_runner.py` (reuses existing baselines + metrics).
- Kahlus-EM Stage 0: `src/neurotwin/em/`.
- Smoke scripts: `scripts/run_dual_field_synthetic.py`, `scripts/run_transition_gym_baselines.py`,
  `scripts/run_ktm_synthetic.py`, `scripts/run_em_artifact_audit.py`,
  `scripts/run_em_passive_logging_analysis.py`.
- Configs: `configs/models/dual_field_synthetic.yaml`, `configs/em/stage0_artifact_audit.yaml`,
  `configs/em/stage1_passive_logging.yaml`.

## Claim boundaries (hard)

- No scientific-superiority or SOTA claim. The shared runner computes no calibration, so the
  evidence gate correctly returns `scientific_claim_allowed=false` for every sweep.
- No `do(a)` / causal language unless an intervention is actually randomized/assigned in data.
- No A100/cluster jobs until local synthetic tasks, baselines, tests, and evidence gates pass.
- No fake results or placeholder "wins"; every artifact is marked synthetic/scaffold/proposed.

## EM safety boundary (hard)

- Stage 0 is **no-human artifact audit** + passive logging only.
- No stimulation, no 20kV equipment, no DBD plasma, no gas canisters, no homemade coils, no
  God Helmet replication, no high voltage, no clinical diagnosis/treatment claim.
- `geomagnetic_fetcher` is **offline only** (never accesses the network).
- `EMContext.field_strength_arb` is an arbitrary synthetic magnitude for differentiating audit
  conditions in simulation — it is never a delivered physical dose.
