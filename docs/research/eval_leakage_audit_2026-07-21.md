# Eval-leakage audit — "are we still making the overlap mistake?"

**Date:** 2026-07-21. **Trigger:** Amrith flagged the historical forecasting
result as invalid (128→127/127 overlapping target). Adversarial read of the
evaluation code, zero tolerance, assume it was written by someone else.
**Verdict:** the overlap mistake is **still live in code**, in two claim-relevant
paths, plus the metric that scores them. Two lower-severity items noted.

The one-line test for this whole class of bug: **a forecasting task is only a
forecast if the target window does not share samples with the input window.** For
`x = signal[a:a+W]`, `y = signal[a+H:a+H+W]`, the input and target overlap by
`W - H` samples whenever `H < W`. A real forecast needs `H >= W`. Nothing in the
repo enforces this.

---

## FINDING 1 — HIGH — prepared/A100 forecasting task is structurally overlapping

**File:** `src/neurotwin/data/prepared_tasks.py:277` (`_future_xy`), feeding
`build_future_forecasting_task_from_windows` → `build_prepared_window_tasks` (the
benchmark suite) and `eval/paper_demos.py`.

```python
def _future_xy(signals):
    usable = [s for s in signals if s.shape[0] >= 2]
    return ([s[:-1] for s in usable],   # x = window[0 : W-1]
            [s[1:]  for s in usable])   # y = window[1 : W]
```

`x` and `y` are the same window shifted by **one sample** → overlap `W-2` of `W-1`
(e.g. 6 of 7 for W=8). There is **no horizon parameter at all** — this path
*cannot* be made non-overlapping without a code change. This is the exact
structure of the invalidated 128→127/127 historical result, in the path that runs
at A100 scale and would produce headline forecasting numbers.

**Proof it produces the illusion:** on real Sleep-EDF, a ridge trained on the
overlapping full-window task scores MSE **0.0018**; the identical ridge on the
isolated single-future-sample task scores **0.227** (126× worse). See
`runs/amrith_isolated_check/REPORT.md` §1.

**Fix:** give the prepared future task a real horizon and emit a `metric_mask`
that scores **only the strictly-future tail** (the `SupervisedWindowTask` already
has a `metric_mask` field — masked-reconstruction uses it; forecasting sets it to
`None`). Minimum: build `y` as `signal[W : W+horizon]` from a longer window, or
mask the metric to the non-overlapping tail. Add an assertion that input/target
share zero indices.

---

## FINDING 2 — HIGH — the M0 "ground-truth ruler" uses a 7/8-overlap task

**File:** `src/neurotwin/forecastability/m0.py:90`

```python
task = build_future_forecasting_task(dataset, window_length=8, forecast_horizon=1, stride=2)
```

`H=1 < W=8` → input/target overlap 7 of 8 samples. The M0 report literally calls
itself *"the ground-truth evaluator"* (`m0.py:182`) and emits a baseline table
that would show persistence/model with near-zero MSE for the copy reason. The
gate that is supposed to *prevent* the mistake **is** the mistake. (Its pass
criteria are only bit-stability + baseline presence, so it does not assert a
scientific win — but anyone reading the emitted MSE table as forecasting skill is
re-fooled.)

**Fix:** `forecast_horizon >= window_length` (e.g. `window_length=8,
forecast_horizon=8`), or isolate the metric to the future tail.

---

## FINDING 3 — HIGH — the metric aggregates over the trivial (copyable) positions

**File:** `src/neurotwin/benchmarks/baseline_suite.py` — `_run_task_models` calls
`_metrics(task.y_test, pred, task.metric_mask, ...)` with `metric_mask=None` for
`future_state_forecasting`, so MSE is averaged over **every** position of the
target window. When 126/127 positions are copyable, the average is dominated by
gimmes and the one genuinely-future position is invisible.

Also `_predict_persistence` (`baseline_suite.py:417`) returns `x_test.copy()` when
shapes match — predict `Y = X` — which on the overlapping task is near-optimal for
a trivial reason, making the persistence baseline look artificially strong (or, as
the model learns the shift, artificially weak vs the model — either way the ranking
is meaningless).

**Fix:** for any forecasting task, set `metric_mask` to the strictly-future tail
so the score reflects only positions absent from the input.

---

## FINDING 4 — MEDIUM — no guard forbids `forecast_horizon < window_length`

**File:** `src/neurotwin/eeg_v1/dataset.py:120` (`build_future_forecasting_task`)
validates `window_length > 1`, `forecast_horizon >= 1`, `stride >= 1` — but **not**
`forecast_horizon >= window_length`. The default `(window_length=8,
forecast_horizon=1)` is overlapping, and `tests/eeg_v1/test_eeg_v1_sprint_a.py:47`
even names a variant `non_overlap` while using `forecast_horizon=2, window_length=8`
— still 6/8 overlap. The author conflated *window stride* (overlap between
consecutive windows) with *input↔target overlap* (horizon vs window length). The
dangerous one is the latter and it is unguarded.

**Fix:** add `if forecast_horizon < window_length: raise ValueError(...)` (or an
explicit `allow_overlap=False` default that must be turned off deliberately), and
correct the mislabeled test.

---

## FINDING 5 — LOW / verify-upstream — subject split is only as safe as `subject_id`

**File:** `src/neurotwin/data/split_manifest.py` — `_group_split` on
`subject_id` is correct and disjoint. The historical person-leakage bug (26
people overlapping train/test) came from **adapters populating `subject_id` from
the night/record**, not from the split logic. The split code is clean; the risk
lives in each dataset adapter's `subject_id` assignment. **Action:** audit
`adapters/multidataset.py` and the Sleep-EDF adapter to confirm two nights of one
person map to one `subject_id`, not two.

---

## My own deliverable script — audited, hardened

`scripts/amrith_isolated_forecast_check.py` (the table going to Amrith):

- **Overlap:** none. `_future_windows`-style targets are strictly future
  (`sig[i+L+h-1]`, `h>=1`); the isolated metric scores one future sample. ✔
- **Subject split:** train/test are disjoint subjects (`name[:5]`), no window
  crosses the boundary. ✔
- **Normalization:** per-recording z-score uses full-slice mean/std (includes
  future). Reviewed: it is a single affine constant per recording applied
  identically to input, target, and every method — it cannot leak the specific
  future value, and mean-of-trace uses context-only stats. Standard practice;
  kept, documented. (Purist alternative: causal/train-only normalization.)
- **Self-test gate:** `--selftest` now asserts (a) persistence beats mean at h=1
  on a random walk, (b) error grows with horizon, (c) a ridge strongly beats the
  mean on AR(1) — proving the learned path isn't silently broken by the spurious
  numpy-2 matmul warnings — and (d) an input-shifted copier scores near-zero on
  the full-window task, i.e. it reproduces the trap on demand. Real runs are
  gated behind this self-test.
- **Fairness fix already applied:** GRU trained directly per horizon (not one-step
  rollout), apples-to-apples with ridge.

---

## Recommended fix order

1. **Findings 1–3 together** (the live claim-relevant path): add a `horizon`
   to the prepared future task, and set `metric_mask` to the strictly-future tail
   so the metric can never again be dominated by copyable positions. One change
   closes the A100 path, the M0 ruler, and the metric.
2. **Finding 4:** the guard — cheapest insurance that this class can't silently
   return. Do it even if 1–3 slip, because the guard *fails the build* the next
   time someone writes an overlapping task.
3. **Finding 5:** adapter `subject_id` audit before any real-data claim run.

Findings 1–3 touch claim-bearing evidence and several tests, so they should land
as a deliberate PR with updated fixtures, not a silent edit.
