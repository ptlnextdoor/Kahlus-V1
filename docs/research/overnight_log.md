# Overnight Log - 2026-07-03

## 00:40 PDT

Hypothesis: the fastest high-trust progress is evidence integrity plus a stricter RFS baseline.  
Action: read attached run spec, AGENTS.md rules, graph report, M0-M3 code/tests, artifacts.  
Result: confirmed stale M3 artifact and available local roots `/tmp/kahlus_chbmit_subset` and `/tmp/kahlus_sleep_edf_subset`.  
Decision: regenerate artifacts from local public-data subsets after code changes.

## 00:47 PDT

Hypothesis: current RFS overstates CHB-MIT because the baseline is weaker than moving average.  
Action: changed `_run_fixture` to select the lowest-NLL gated baseline before computing RFS.  
Result: M3 CHB-MIT primary RFS changed from a small positive value to `-0.388076` bits against moving average.  
Decision: keep this as the hardened-RFS improvement.

## 00:50 PDT

Hypothesis: the residual head should be a convex offset-GLM, not a hand-rolled clipped GD path.  
Action: checked opensrc paths:

- `/Users/aayu/.opensrc/repos/github.com/scikit-learn/scikit-learn/main/sklearn/linear_model/_logistic.py`
- `/Users/aayu/.opensrc/repos/github.com/scipy/scipy/main/scipy/optimize/_minimize.py`

Result: scikit-learn has regularized logistic solvers but no offset API; SciPy is not a project dependency.  
Decision: implement minimal IRLS offset logistic locally, no new dependency.

## 00:53 PDT

Hypothesis: the novel contribution should be a horizon-wise RFS curve.  
Action: added `M4` with patient-safe horizon labels, synthetic known/null curves, and optional Sleep-EDF smoke.  
Result: synthetic known curve decays across horizons; synthetic null stays near zero. Sleep-EDF smoke runs but fails for underpower/time-shift control.  
Decision: report M4 as a benchmark-method gate with honest real-smoke failure.

## 00:55 PDT

Hypothesis: stale artifacts need an executable guard.  
Action: added M3 artifact freshness test that recomputes gate failures from current logic and requires `tusz_external` schema.  
Result: test failed on stale artifact, then passed after regeneration.  
Decision: keep focused guard instead of broad M0 byte-equality because M0 records git state and is expected to change with dirty worktrees.

## 00:56 PDT

Hypothesis: repo needs minimum CI.  
Action: added `.github/workflows/ci.yml` with ruff, pytest, and `git diff --check`.  
Result: local forecastability tests pass.  
Decision: run full test/ruff after docs.
