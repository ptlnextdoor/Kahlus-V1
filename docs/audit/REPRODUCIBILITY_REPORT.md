# Reproducibility Report

## What Can Be Reproduced

- Unit and integration tests can be run from source with an explicit `PYTHONPATH=src`.
- Synthetic forecastability reports and many evidence-contract artifacts have deterministic local commands.
- The recovered A100 release contains checksummed summaries, configs, logs, and rank metrics sufficient to inspect its finalization claim.

## What Cannot Be Reproduced Independently

- The 100,000-step training trajectory, because the release asset omits the referenced checkpoints and complete raw-input identities.
- Exact MOABB preprocessing, because all dependency-resolved paradigm defaults, transforms, and raw source hashes are not frozen.
- A paper-ready model comparison, because baseline tuning, seeds, compute, and model-selection budgets are not equivalent.
- Clinical or cross-dataset claims, because no qualifying evidence exists.

## Environment Risks

Running tests without `PYTHONPATH=src` imported a stale editable `neurotwin` package from another worktree and produced collection errors. This is a reproducibility defect: commands must prove which source tree was imported. The local and A100 PyTorch/CUDA environments also differ, and the project lacks a full lockfile or immutable container digest.

With the correct source path, the full audit run completed with 751 tests passed, 3 skipped, 114 subtests passed, and 1 failure. The sole failure was a live PhysioNet `RECORDS` request in `test_sleep_edf_records_index_source` timing out. A default test suite that depends on external network availability is not hermetic.

## Minimum Reproduction Contract

Every evidence bundle must include:

- source commit and clean-worktree proof;
- immutable container digest or complete lockfile;
- Python, CUDA, driver, framework, and library versions;
- canonical dataset URLs, versions, licenses, and raw-file hashes;
- split manifest generated before windowing;
- preprocessing DAG with train-fitted parameters;
- task sample-range manifest proving no context-target overlap;
- full config and command line;
- random seeds and deterministic-mode status;
- checkpoints needed for evaluation/resume or an explicit exclusion statement;
- per-subject metrics, failed runs, and gate reports;
- audit script that verifies hashes and imports the bundled source.

## One-Command Standard

A clean environment should be able to run `prepare -> audit -> train/evaluate -> aggregate -> report` without editing local paths, downloading from mirrors of uncertain provenance, or relying on files outside the declared data cache.

## Historical Result Policy

Changing the target definition invalidates comparison with historical MSE values. Old artifacts must remain immutable and receive an explicit `overlapping_target_task` label; they must not be silently regenerated under the same result identifier.

## Audit Verification Record

- `PYTHONPATH=src /opt/miniconda3/bin/python -m pytest -q`: 751 passed, 3 skipped, 114 subtests passed, 1 live-network timeout failure in 538.02 seconds.
- `PYTHONPATH=src /opt/miniconda3/bin/python -m pytest -q tests/forecastability/test_m2.py -k 'not sleep_edf_records_index_source'`: 6 passed, 1 deselected.
- `ruff check .`: passed.
- Shell syntax checks for tracked runner/cluster/Slurm shell scripts: passed.
- `git diff --check`: passed.
- Claim-matrix JSON syntax, required-field coverage, status enum, and 17-record count: passed.
