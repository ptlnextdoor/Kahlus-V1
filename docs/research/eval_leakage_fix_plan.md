# Remediation plan — close the overlap/leakage findings

**Status (2026-07-21):** Phase 0 complete · Phase 1 (PR-A) implemented on
`fix/forecast-overlap-metric` — **stop for human review before Phase 2.**
Phases 2–3 not started.

**Source:** `docs/research/eval_leakage_audit_2026-07-21.md` (findings 1–5).
**Shape:** three PRs, in order, with a review + truthfulness gate between each. Do
**not** batch them. Each phase ends green (tests pass) and honest (no emitted
number left standing that the fix invalidated).

**Non-negotiable guardrail (carried from `kahlus-honest-posture`):** a fix that
changes any forecasting number must (a) regenerate every artifact/doc that quoted
the old number and (b) mark the old number invalid in place — never silently
overwrite. A null or "smaller than we thought" result is a valid outcome.

---

## Core design decision (applies to PR-A)

There are two ways to make a forecast honest; pick the **mask** approach as
primary because it is contained and reversible, and it fixes the *reported* number
even where the window still overlaps:

- **(chosen) Strictly-future mask.** A forecasting `SupervisedWindowTask` carries a
  boolean `metric_mask` (the field already exists and masked-reconstruction already
  uses it end-to-end) that is `True` only on target positions **absent from the
  input window**. Apply that mask to **both the training loss and the eval metric**.
  For the current shift-by-one windows (`y=w[1:]`, `x=w[:-1]`) exactly one position
  per window is strictly future (the last) — masking to it reproduces the isolated
  single-sample forecast, and the copy can no longer inflate the score.
- **(noted alternative) Disjoint windows.** Restructure so `x=w[:-H]`, `y=w[-H:]`
  (input and target share zero indices). Cleaner semantically but changes array
  shapes and ripples into every model's forecast head — larger blast radius. Keep
  as a follow-up only if the mask approach proves insufficient.

Both are consistent with Finding 4's guard: the guard will *require* that a
forecasting task either has `forecast_horizon >= window_length` **or** carries a
strictly-future `metric_mask`. So the mask approach is what makes the guard
satisfiable without breaking the `(8,1)` default.

---

## Phase 0 — branch + baseline capture (no code change)

1. Branch from `main` (not the current `add-researchdock-roadmap`):
   `git checkout main && git pull && git checkout -b fix/forecast-overlap-metric`.
2. Capture the *current* state so the change is provable and truthful:
   - Run the full test suite, save pass/fail counts to `runs/audit_baseline/tests_before.txt`.
   - Run the M0 gate and the prepared suite once; archive the emitted
     `baseline_table.csv` / `M0_EVIDENCE_REPORT.md` as `*_BEFORE` (these hold the
     inflated numbers we are about to invalidate — keep them as evidence of the bug,
     labeled invalid).
   - `grep -rn` the repo/docs for any quoted forecasting MSE/`pearsonr`/`r2` so we
     know every place that must be regenerated or marked invalid in Phase 1 step 5.
3. Definition of done: baseline captured, nothing changed yet.

## Phase 1 — PR-A: Findings 1, 2, 3 (the live overlap + metric)

**Files:** `data/prepared_tasks.py`, `benchmarks/baseline_suite.py`,
`eeg_v1/dataset.py`, `forecastability/m0.py`, `training/prepared_metrics.py`
(+ their tests).

1. **Add the strictly-future mask at task construction.**
   - `prepared_tasks.py` `build_future_forecasting_task_from_windows` /
     `_future_xy`: compute `metric_mask` marking the target positions not present in
     the input, set it on the returned `SupervisedWindowTask`. Give `_future_xy` a
     `horizon` param (default 1) instead of the hardcoded `[1:]/[:-1]`, so the
     shape and the mask are derived from one place.
   - `eeg_v1/dataset.py` `build_future_forecasting_task`: same — attach a
     strictly-future `metric_mask` to its `SupervisedWindowTask.metadata`/field.
2. **Score only the mask.**
   - `benchmarks/baseline_suite.py` `_run_task_models` → `_metrics(...)`: already
     receives `task.metric_mask`; verify it applies it for forecasting (it does for
     masked-recon). The change is just that forecasting now *has* a mask.
   - `training/prepared_metrics.py` `evaluate_task` / `mse_loss`: apply the same
     mask to the **training loss** so the model is optimised for the future tail,
     not the copyable positions.
3. **M0 ruler:** `forecastability/m0.py:90` — either bump to
   `forecast_horizon=window_length` or rely on the new mask; assert in the M0 code
   that the task it runs carries a strictly-future mask (fail loud otherwise).
4. **Tests (new + updated):**
   - New `tests/.../test_forecast_no_overlap.py`: build the forecasting task, assert
     input and every metric-scored target position share **zero** sample indices;
     assert a copy-input-shifted predictor does **not** get a near-zero *masked*
     score (the regression test for this exact bug).
   - Update `tests/eeg_v1/test_eeg_v1_sprint_a.py` (fix the mislabeled `non_overlap`
     case), `tests/benchmarks/test_prepared_event_suite.py`,
     `tests/eval/test_audit_prepared.py`, `tests/scoring/test_metrics.py`,
     `tests/data/test_multidataset_a100.py`, forecastability tests.
5. **Truthfulness pass (blocking):**
   - Regenerate M0 + prepared artifacts; diff against the `*_BEFORE` archive.
   - For every quoted forecasting number found in Phase 0 step 2c: regenerate it, or
     if it was claim-bearing and is now invalid, mark it invalid in place with a
     dated note pointing at this PR. Do **not** delete history.
   - Re-run `scripts/amrith_isolated_forecast_check.py --selftest` and the real run;
     confirm the masked prepared-path number now matches the isolated-metric story
     (they should agree — that agreement is the proof the fix is real).
6. **Green gate:** full test suite passes; `tests_after.txt` vs `tests_before.txt`
   diff explained (every changed test has a one-line reason).
7. Open PR-A. **Stop. Human review before Phase 2.**

## Phase 2 — PR-B: Finding 4 (the guard)

Branch `fix/forecast-overlap-guard` from `main` **after PR-A merges** (it depends on
the mask existing).

1. In `eeg_v1/dataset.py build_future_forecasting_task` and the prepared builder:
   add validation — a forecasting task is only valid if
   `forecast_horizon >= window_length` **or** it carries a non-empty strictly-future
   `metric_mask`. Raise `ValueError` otherwise. This *fails the build* the next time
   someone writes an overlapping task with no mask.
2. Test: `test_forecast_overlap_guard.py` — constructing an overlapping task with no
   mask raises; a masked or `H>=W` task passes.
3. Green gate + PR-B. Stop for review.

## Phase 3 — PR-C: Finding 5 (adapter subject_id audit)

Branch `audit/adapter-subject-id` from `main` after PR-B.

1. Read `adapters/multidataset.py` + the Sleep-EDF adapter path; trace how
   `subject_id` is assigned. Confirm two nights of one person → one `subject_id`
   (the historical bug was nights-as-identities → 26-person train/test overlap).
2. Add a split-audit assertion/test on a real (or realistic fixture) manifest: no
   `subject_id` appears in more than one of train/val/test; and the count of unique
   people is < count of recordings when multi-night data is present (catches the
   nights-as-identities regression).
3. If a real leak is found, fix the adapter mapping; regenerate any affected split
   manifests. Truthfulness pass as in Phase 1 step 5.
4. Green gate + PR-C.

---

## Cross-cutting rules
- One finding-group per PR; never mix. Each PR merges to `main` green before the
  next branches.
- Every PR description states: what number moved, why, and where the old number is
  now marked invalid.
- No new dependencies; reuse existing `metric_mask`, `SplitManifest`, metrics.
- If any fix makes a former "positive" forecasting result disappear, that is the
  correct outcome — report it plainly, do not re-tune to recover it.

## Test commands (per phase)
```bash
PYTHONPATH=src python3 -m pytest -q            # full suite
PYTHONPATH=src python3 -m neurotwin.cli ...    # M0 / prepared regen (Phase 0 & 1)
PYTHONPATH=src python3 -W ignore scripts/amrith_isolated_forecast_check.py --selftest
```

## Definition of done (whole effort)
- Overlapping input/target can no longer produce a reported forecasting score
  (masked metric + loss).
- The build fails if anyone writes an overlapping forecasting task without a
  strictly-future mask.
- Subject splits provably group multi-night people correctly.
- Every previously-emitted forecasting number is either regenerated or marked
  invalid in place; nothing silently changed.
