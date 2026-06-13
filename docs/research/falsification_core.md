# Falsification core (`src/neurotwin/falsification.py`)

Shared, **lane-neutral** harness for the synthetic v-lane falsification benchmarks (v2, v3, and
future EM/v2.5). Extracted in Sprint 1B.5 so each lane stops hand-copying the same Outcome type,
finite check, gate assembly, and report/write boilerplate.

> **Lane neutrality is a hard rule.** This module imports only stdlib, `numpy`,
> `neurotwin.gates`, and `neurotwin.repro`. It must **never** import a lane (`models/dual_field`,
> `transition_gym`, `em`, `models/ktm`). Depending on the core therefore never couples lanes to
> each other — they each depend on a neutral base.

## API

### `Outcome(name: str, passed: bool, detail: dict, reason: str = "")`
One diagnostic result. `detail` holds numeric scores (may nest dicts). `reason` is a
human-readable failure message (empty when passed). Lanes alias it for local readability
(`DiagnosticOutcome = Outcome`, `V3Outcome = Outcome`).

### `outcomes_finite(outcomes) -> bool`
True iff every numeric value in every outcome `detail` — **including nested dicts/lists** — is
finite. Bools and strings are ignored. (Replaces two divergent per-lane walkers; the old v2 one
silently skipped nested dicts.)

### `assemble_gate(*, branch, dataset, claim_scope, outcomes, required, split_audit_passed=True, baseline_table_present=True, extra_finite=True) -> dict`
Builds the unified evidence gate. The narrow claim is calibrated by the `required` diagnostics:
`calibration_checked = all(required pass)`, and each failed required outcome becomes a gate
failure reason. `finite_metrics = outcomes_finite(outcomes) and extra_finite` (pass
`extra_finite` to fold in finiteness of inputs outside the outcomes, e.g. a baseline leaderboard).
Delegates to `neurotwin.gates.evaluate_gate`, so the output is the exact dossier schema and the
claim is blocked unless `claim_scope` is in the narrow allowlist.

### `build_report(*, schema, branch, claim_scope, seed, config, outcomes, gate, extra=None) -> dict`
Standard report dict: schema/branch/claim_status/claim_scope/seed/config/diagnostics +
`falsification_passed`/`scientific_claim_allowed`/`failure_reasons`/`evidence_gate` derived from
`gate`. `extra` merges lane-specific keys (e.g. v3's `baseline_leaderboard`, `ranking`,
`ktm_beats_baselines`).

### `write_report(out_dir, *, report, gate, prefix) -> {"report": Path, "evidence_gate": Path}`
Writes `{prefix}_benchmark_report.json` + `{prefix}_evidence_gate.json`.

## How a lane wires a falsifier

```python
from neurotwin.falsification import Outcome, assemble_gate, build_report, write_report

def my_diagnostic(data) -> Outcome:
    score = ...                      # compute on a leakage-safe split
    ok = score >= THRESHOLD
    return Outcome("my_diagnostic", ok, {"score": float(score)},
                   "" if ok else f"score {score:.3f} below {THRESHOLD}")

def run_my_benchmark(config, *, seed=None):
    outcomes = [my_diagnostic(d) for d in ...]
    gate = assemble_gate(
        branch="em", dataset="em_stage0", claim_scope="em_artifact_audit_no_human",
        outcomes=outcomes, required=[o.name for o in outcomes],  # or a gate-critical subset
    )
    return Result(outcomes=outcomes, gate=gate,
                  passed=bool(gate["scientific_claim_allowed"]),
                  failure_reasons=list(gate["failure_reasons"]))

def benchmark_report(result):
    return build_report(schema="kahlus.em_stage0.v1", branch="em",
                        claim_scope="em_artifact_audit_no_human", seed=result.seed,
                        config=result.config.__dict__, outcomes=result.outcomes, gate=result.gate)
```

Add the new `claim_scope` to `NARROW_CLAIM_SCOPES` in `src/neurotwin/gates/unified_gate.py`, or
the gate blocks it as "too broad".

## Canonical baselines

Baseline models live in `src/neurotwin/baseline_runner.py` (`run_baselines`, the
`_run_single_model` dispatch, `regression_metrics`, `retrieval_knn_predict`). A leaderboard should
be a single `run_baselines(...)` call — register new baselines in the dispatch rather than scoring
them inline in a benchmark.

## Reference implementations
- v2: `src/neurotwin/models/dual_field/benchmark.py` + `diagnostics.py`
- v3: `src/neurotwin/transition_gym/benchmark.py` + `operator_recovery.py`
- core tests: `tests/test_falsification.py`
