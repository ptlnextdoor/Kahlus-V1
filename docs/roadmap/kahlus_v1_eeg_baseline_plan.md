# Kahlus v1 EEG Baseline Plan

## What v1 Is

Kahlus v1 is the EEG forecasting baseline lane:

```text
EEG patch/window tokenizer
-> temporal encoder
-> latent state z_t
-> future-window decoder
```

This sprint makes the lane benchmarkable under subject-held-out splits with strong local
baselines and an explicit evidence gate.

## What v1 Is Not

This sprint does not claim diagnosis, treatment, epilepsy detection, depression detection,
foundation-model status, SOTA, Kahlus v2 success, or Kahlus v3 success. ResearchDock and
Kahlus-Affect remain later device branches and are not implemented here.

## Objective

The v1 training objective is documented as:

```text
L_v1 = lambda_1 MSE(Y_hat, Y)
      + lambda_2 MAE(Y_hat, Y)
      + lambda_3 L_smooth
```

with:

```text
L_smooth = sum_t ||Y_hat_{t+1} - 2Y_hat_t + Y_hat_{t-1}||_2^2
```

The Sprint A implementation exposes the smoothness term for tests and reports the baseline
metrics needed before any main-model claim.

## Baseline Ladder

The default runnable ladder is:

```text
persistence
linear_ridge
autoregressive_ridge
MLP
TCN
Transformer
SSM fallback
NeuralStateSpaceTranslator ("neurotwin")
```

All requested models receive the same window length, forecast horizon, split, and training
step budget. Baselines are results, not scaffolding.

## Split Rule

Subject-held-out is the default. The v1 audit writes `split_audit.json` with train/val/test
subjects, subject-overlap status, window-overlap status, and failure reasons. Session-held-out
is allowed by the gate shape but is not the default script path.

## HBN-EEG External Path

No public dataset is downloaded automatically. The HBN path is local-only and expects a
user-provided `manifest.jsonl` under `--data-root`; each row points to a `.npy` or `.npz`
signal file shaped `[time, channels]`. Missing local data fails clearly instead of falling
back to synthetic data.

## A100 Scaling Boundary

Sprint A/B are local validation sprints only. They must not launch A100 or cluster jobs.
The 7x NVIDIA A100 80GB cluster is reserved for later work after local synthetic fixtures,
subject-held-out audits, baseline ladders, and evidence gates pass from a clean merge/tag.

Future A100 handoff packaging must label the hardware honestly as `7xA100`, include exact
commit hash, clean-worktree proof, runner tarball, configs, checksum manifest, CPU smoke,
DDP/torchrun command, evidence bundle writer, and audit script, and must exclude secrets,
checkpoints unless explicitly allowed, and raw participant data.

## Outputs

Each script run writes:

```text
metrics.json
metrics.csv
baseline_table.json
baseline_table.csv
split_audit.json
evidence_gate.json
run_config.json
dataset_summary.json
failure_reasons.json
diagnostic_report.md
per_subject_metrics.csv
per_channel_metrics.csv
per_horizon_metrics.csv
baseline_verification.json
baseline_checksum_manifest.json
```

The narrow allowed claim scope is `eeg_future_forecasting_benchmark_ready`.

## Verification

```bash
PYTHONPATH=src python3 scripts/run_eeg_v1_baselines.py --dataset synthetic_fixture --out-dir /tmp/kahlus_v1_eeg_smoke --seed 0
```

## Sprint C: Reviewer-Facing Autocorrelation Summary

The baseline smoke command now prints the short-horizon, shuffled-target, long-horizon,
and non-overlapping-window diagnostic MSE summary fields directly to stdout. The Markdown
diagnostic report includes the same summary table so reviewers can see why a low ridge or
persistence MSE may reflect autocorrelation rather than model understanding.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint A.3: First-Class SSM and Train-Only Shuffled Control

The EEG v1 smoke now treats `tiny_ssm` as the first-class SSM baseline in the default ladder. The
older `ssm_fallback` executable remains available for compatibility, but the narrow
`eeg_future_forecasting_benchmark_ready` artifact gate requires `tiny_ssm` to be present in emitted
metrics.

The shuffled-target negative control is also required. It shuffles training targets only, keeps
validation/test targets unchanged, and records `shuffle_boundary=train_split_only` in the
autocorrelation diagnostic row. This prevents the control from proving anything by corrupting
held-out targets.

The diagnostic report now surfaces `tiny_ssm_mse`, `shuffled_target_control_mse`,
`persistence_or_ridge_dominates`, and `shuffled_target_close_to_real_baselines` so reviewers can see
whether low MSE is still dominated by simple continuation or broken target alignment.

Verification:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/eeg_v1 -v
PYTHONPATH=src python3 scripts/run_eeg_autocorr_diagnostics.py --dataset synthetic_fixture --out-dir /tmp/kahlus_v1_a3_autocorr --seed 0
```

## Sprint D: HBN Local Manifest Validation

The HBN-EEG local adapter now rejects relative manifest paths that escape `--data-root` and
rejects unsupported signal file extensions before NumPy loading. Absolute paths remain allowed
only because the adapter contract already permits user-provided local absolute paths; this is
local validation, not an automatic public-data loader or downloader.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint E: HBN Local Provenance Boundary

HBN-style local manifest runs now write `data_source` and `benchmark_status` into stdout,
`run_config.json`, `metrics.json`, and the diagnostic report. Local manifest fixture runs are
explicitly marked `local_manifest_not_public_hbn_benchmark` so fixture smoke results cannot be
presented as public HBN benchmark evidence.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint F: HBN Numeric Manifest Validation

The HBN-EEG local adapter now rejects non-finite or non-positive `sampling_rate` values before
time-vector construction and rejects signal arrays containing NaN or Inf before any windowing,
baseline, or report path can run. This is local-manifest validation only; it does not add public
dataset downloads, loaders, or benchmark claims.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint G: HBN Subject Metadata Validation

The HBN-EEG local adapter now rejects manifest rows with missing or blank `subject_id` values
before split-manifest construction. Subject identity is required evidence for subject-held-out
claims, so malformed local rows fail clearly instead of producing raw key errors or empty-subject
splits.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint H: HBN Manifest Row Parsing Validation

The HBN-EEG local adapter now rejects malformed JSONL rows and non-object JSON rows with
line-numbered `ValueError` messages before path, signal, or split logic runs. This keeps local
fixture failures auditable without adding any public-data loader or benchmark claim.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint I: HBN Nonempty Signal Shape Validation

The HBN-EEG local adapter now rejects `[time, channels]` arrays with an empty time axis or empty
channel axis before time-vector, split, or baseline logic runs. Forecasting windows require at
least one sample and one channel, so empty local fixtures fail clearly instead of becoming
misleading benchmark artifacts.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint J: HBN Channel Count Consistency Validation

The HBN-EEG local adapter now rejects local manifests whose recordings have inconsistent channel
counts. This deliberately fails closed until explicit montage/channel-alignment metadata exists,
because silently mixing different channel dimensions would make baseline comparisons ambiguous.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint K: HBN Unique Record Identity Validation

The HBN-EEG local adapter now rejects duplicate `record_id` values before split manifests,
metrics, or evidence artifacts are built. Default generated IDs remain unique by row index; this
guard protects user-provided local manifest identifiers from silently aliasing two recordings.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint L: HBN Optional Channel Names Validation

The HBN-EEG local adapter now validates optional `channel_names` metadata when local manifests
provide it. Names must be a JSON array of nonempty strings, match the signal channel count, and
stay consistent across recordings; otherwise the adapter fails closed rather than silently
pretending channel alignment is known.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint M: HBN Optional Text Metadata Validation

The HBN-EEG local adapter now rejects blank provided `session_id`, `site_id`, and `record_id`
values. Omitted optional fields still receive deterministic defaults, but explicitly blank fields
fail closed so local evidence artifacts do not carry ambiguous recording metadata.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint N: HBN Sampling Rate Consistency Validation

The HBN-EEG local adapter now rejects local manifests whose recordings use mixed
`sampling_rate` values. This fails closed until an explicit resampling path exists, because sample
windows at different time scales should not be compared as equivalent baseline evidence.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint O: Dataset Summary Evidence Artifact

The EEG v1 baseline runner now writes `dataset_summary.json` with bounded, non-raw metadata:
dataset/source/status, split type, subject counts, train/val/test window counts, window settings,
and channel count. This gives reviewers a compact evidence check without exposing raw signals or
inflating benchmark claims.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint P: Invalid Task Configuration CLI Guard

The EEG v1 baseline runner now reports impossible forecast task settings as a clear local
configuration error instead of a Python traceback. Invalid window geometry is not a model result
and must fail before baseline tables, evidence gates, or interpretation artifacts are written.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint Q: Baseline Evidence Gate Criteria

The EEG v1 baseline evidence gate now records its decision criteria directly in
`evidence_gate.json`: minimum forecast horizon, allowed held-out split types, required split
audit, baseline table, finite metrics, calibration check, and the allowed narrow claim scope.
The diagnostic report renders the same criteria so reviewers can see that the gate is
benchmark-readiness only, not a model-understanding, clinical, or SOTA threshold.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint R: Baseline Report Run Config Summary

The EEG v1 baseline diagnostic report now includes a `Run Config` section sourced from the
same bounded metadata written to `run_config.json`: seed, train-step budget, model list,
window settings, data source/status, selection policy, and claim scope. This makes the
human-readable report replayable without opening the JSON sidecar first.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint S: Baseline Report Split-Audit Failure Summary

The EEG v1 baseline diagnostic report now includes a `Split Audit Failures` section when the
split audit records detailed leakage or split-validation failures. This surfaces the underlying
split evidence in the human-readable report instead of only showing the derived gate failure.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint T: Baseline Failure Reasons Split-Audit Sidecar

The EEG v1 baseline `failure_reasons.json` artifact now includes `split_audit_failures`
alongside baseline and gate failures. This gives audit consumers the same structured
split-failure sidecar coverage already present in the few-shot adaptation evidence bundle.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint U: Baseline Report Baseline-First Method Order

The EEG v1 baseline diagnostic report now includes a `Method Order` table before ranking.
Requested methods are tagged as `baseline` or `main_model`, so low-MSE baseline results are
visible before any comparison with NeuroTwin/Kahlus.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint V: Target Scale Context for Low MSE Interpretation

The EEG v1 baseline bundle now writes `target_scale_context.json` and a matching diagnostic
report section. It reports target mean/std/variance/min/max and model RMSE/MSE ratios against
target scale, making clear that synthetic fixture MSE is in normalized fixture units rather
than raw EEG microvolts.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint W: Target Scale Context in Smoke Stdout

The EEG v1 baseline runner now prints target units, target standard deviation, target variance,
and the best baseline RMSE relative to target standard deviation directly in stdout. The smoke
command therefore explains low normalized MSE without requiring the reviewer to open the JSON
sidecar first.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint X: Metric Breakdown Summary in Diagnostic Report

The EEG v1 diagnostic report now summarizes how many per-subject, per-channel, and per-horizon
metric rows were written, and points to the detailed CSV sidecars. This keeps the Markdown report
auditable while leaving detailed rows in machine-readable files.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint Y: Baseline Gap Summary in Diagnostic Report

The EEG v1 diagnostic report now renders the existing persistence, ridge, and best-baseline gap
metrics with an explicit sign convention. This makes baseline wins or losses visible in the
Markdown report instead of requiring reviewers to inspect `metrics.json`.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint Z: Baseline Failure Details in Diagnostic Report

The EEG v1 diagnostic report now lists requested baseline failures from the same payload written
to `failure_reasons.json`. Unknown or unavailable baselines remain failures; the Markdown report
now exposes them directly for audit.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint AA: Delta Prediction Summary Exposure

The EEG v1 autocorrelation diagnostics already run a delta-prediction control where the target is
future-minus-input instead of the raw future signal. The summary, CLI stdout, and diagnostic
report now expose the delta-control best MSE gap against the short-horizon raw task so reviewers
can see whether low raw MSE is just waveform continuation.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint AB: Autocorrelation Diagnostic Reasons in Report

The EEG v1 diagnostic report now includes a reason column for each autocorrelation diagnostic.
Skipped, blocked, or not-applicable controls are therefore visible in the human-readable report
instead of only in `autocorrelation_diagnostics.csv`.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint AC: HBN Sampling Rate Provenance

The v1 future-window task now carries a single finite dataset sampling rate into `run_config.json`,
`dataset_summary.json`, and the diagnostic Markdown report. This makes local HBN-style fixture
windows auditable in seconds/Hz terms after the adapter has already rejected mixed-rate manifests.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint AD: Baseline Evidence Checksum Manifest

The EEG v1 baseline artifact bundle now writes `baseline_checksum_manifest.json` with SHA-256
digests and byte counts for the emitted JSON/CSV/Markdown evidence artifacts. The manifest
excludes itself to avoid circular hashing and gives reviewers a local integrity check before
trusting baseline, autocorrelation, split-audit, or gate evidence.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint AE: Baseline Checksum Audit Script

The EEG v1 baseline lane now includes `scripts/audit_eeg_v1_baseline_checksums.py`, a local
JSON-emitting verifier for `baseline_checksum_manifest.json`. It fails closed on missing manifests,
invalid JSON, unsupported schemas or algorithms, invalid artifact rows, duplicate paths, missing
artifacts, byte-count changes, and checksum mismatches, so the baseline checksum manifest becomes
a runnable evidence gate.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint AF: Baseline Checksum Audit Command in Smoke Stdout

The EEG v1 baseline runner now prints the `baseline_checksum_manifest.json` path and a concrete
`checksum_audit_command` using the actual `--out-dir` value from the smoke run. This makes the
baseline checksum gate copy/pasteable from stdout while preserving the local-only lane and leaving
training, ranking, gates, and claim scope unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint AG: Required Baseline Artifact Manifest Coverage

The EEG v1 baseline checksum audit now requires `baseline_checksum_manifest.json` to include every
core baseline evidence artifact that the writer always emits: metrics, tables, split audit, gate,
run config, dataset summary, target-scale context, failure reasons, report, and per-subject,
per-channel, and per-horizon metrics. Optional autocorrelation sidecars remain outside this sprint's
required-entry contract because direct writer calls can omit that diagnostic payload.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint AH: Unexpected Baseline Checksum Manifest Entry Rejection

The EEG v1 baseline checksum audit now rejects validly check-summed but unexpected entries in
`baseline_checksum_manifest.json`. The allowlist is bounded to the required core evidence artifacts
plus optional autocorrelation diagnostic JSON/CSV sidecars, so a manifest cannot silently bless extra
files outside the declared local baseline evidence bundle.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint AI: Baseline Verification Sidecar

The EEG v1 baseline artifact bundle now writes `baseline_verification.json`, a machine-readable
sidecar declaring the local-only execution lane, `a100_jobs_launched=false`, the checksum manifest
name, and the exact checksum-audit command for the emitted artifact directory. The sidecar is covered
by `baseline_checksum_manifest.json`, so local verification instructions are auditable evidence
rather than stdout-only prose.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint AJ: Baseline Verification Sidecar Contract Audit

The EEG v1 baseline checksum audit now validates `baseline_verification.json` semantically in
addition to checking its SHA-256 digest. The audit fails if the sidecar no longer declares the
local-only execution lane, `a100_jobs_launched=false`, the expected checksum manifest, or the exact
checksum-audit command for the artifact directory.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint AK: Baseline Report Artifact Index and Checksum Instructions

The EEG v1 baseline diagnostic report now includes an `Artifact Index` table and a `Checksum Audit`
section near the top of the report. The index names the bounded JSON/CSV/Markdown evidence artifacts,
including `baseline_verification.json` and `baseline_checksum_manifest.json`, while the checksum
section gives the local audit command with an explicit `<artifact-dir>` placeholder.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint AL: Baseline Verification Sidecar in Smoke Stdout

The EEG v1 baseline runner now prints the `baseline_verification.json` path and
`a100_jobs_launched=false` directly in smoke stdout. This keeps the one-command local smoke output
connected to the checksum-covered verification sidecar without changing model training, ranking,
gates, or claim scope.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint AM: Stimulus/Task Label Split Audit Diagnostic

The EEG v1 autocorrelation diagnostic now audits stimulus/task label overlap when labelled
recording metadata exists. It reports observed label keys, train/val/test label sets, overlap
status, and failure reasons for the current split; it does not invent a new stimulus-held-out
split builder or claim a labelled public benchmark result.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint AN: Stimulus/Task Split Audit Report Visibility

The EEG v1 diagnostic report now adds a dedicated `Stimulus/Task Split Audit` section when
labelled stimulus/task metadata was audited. The section surfaces observed label keys,
train/val/test label sets, overlap status, leakage status, and failure reasons in Markdown while
preserving the existing JSON sidecar and generic autocorrelation diagnostics table.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint AO: Stimulus/Task Split Audit Evidence Gate Blocker

When labelled stimulus/task metadata exists and the stimulus/task split audit detects label overlap,
the EEG v1 artifact writer now adds a diagnostic gate failure and blocks the narrow
benchmark-readiness claim. Unlabelled synthetic fixture runs remain unchanged; this only tightens
the evidence gate when a completed labelled split diagnostic proves overlap.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint AP: Failure Reasons Sidecar Contract Audit

The EEG v1 baseline checksum audit now validates the structure of `failure_reasons.json`, requiring
`baseline_failures`, `gate_failures`, `split_audit_failures`, and `diagnostic_failures` to exist as
lists. This prevents a checksum-updated bundle from silently dropping diagnostic gate-failure
evidence while still keeping the audit local and schema-focused.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint AQ: Failure Reasons Gate Consistency Audit

The EEG v1 baseline checksum audit now cross-checks `failure_reasons.json` against
`evidence_gate.json` by requiring `failure_reasons.gate_failures` to exactly match
`evidence_gate.failure_reasons`. This prevents checksum-updated evidence bundles from fabricating or
dropping gate failures in the human-facing failure sidecar.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint AR: Failure Reasons Split-Audit Consistency Audit

The EEG v1 baseline checksum audit now cross-checks `failure_reasons.json` against
`split_audit.json` by requiring `failure_reasons.split_audit_failures` to exactly match
`split_audit.failure_reasons`. This prevents checksum-updated evidence bundles from hiding split
audit failures in the human-facing failure sidecar.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a -v
```

## Sprint A.4: Shuffled-Control Separation Gate

The EEG v1 evidence gate now treats the train-only shuffled-target negative control as a required
separation check, not just a reported diagnostic. The narrow
`eeg_future_forecasting_benchmark_ready` claim is blocked when the shuffled-target control does not
degrade or when it remains too close to the real short-horizon baseline performance.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_a4_gate_fails_when_shuffled_target_control_stays_too_close -v
```

## Sprint A.5: Model-Win Claim Status

The EEG v1 evidence bundle now separates benchmark-readiness from a model-performance win. A run can
pass the narrow benchmark-readiness gate while `model_win_claim_allowed` remains false when
persistence/ridge/autoregressive baselines dominate or Kahlus does not beat the best baseline.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_script_writes_expected_artifacts -v
```

## Sprint A.6: Model-Win Checksum Audit Consistency

The EEG v1 baseline checksum audit now cross-checks `model_win_claim_allowed`,
`model_win_status`, and `model_win_claim_failure_reasons` between `metrics.json` and
`evidence_gate.json`. This prevents a checksum-updated evidence bundle from silently changing the
human-facing metrics artifact into a model-win claim while the gate artifact still blocks it.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_model_win_metrics_gate_mismatch -v
```

## Sprint A.7: Autocorrelation Gate Checksum Consistency

The EEG v1 baseline checksum audit now cross-checks the gate-driving shuffled-control fields in
`autocorrelation_diagnostics.json` against `evidence_gate.json`. A checksum-updated diagnostics
artifact cannot mark shuffled targets unsafe unless the gate also records the matching failure.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_autocorr_gate_mismatch -v
```

## Sprint A.8: Passing-Gate Autocorrelation Manifest Requirement

The EEG v1 baseline checksum audit now requires `autocorrelation_diagnostics.json` to be listed in
`baseline_checksum_manifest.json` whenever `evidence_gate.json` allows the benchmark-readiness claim
and declares required negative controls. Blocked gates may still omit optional diagnostics.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_requires_autocorr_manifest_entry_when_gate_passes -v
```

## Sprint A.9: Required Negative-Control Row Audit

The EEG v1 baseline checksum audit now verifies that a passing gate's required negative controls are
present as completed rows inside `autocorrelation_diagnostics.json`. A diagnostics file cannot keep
its checksum valid while dropping `shuffled_target_control`.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_missing_required_autocorr_control -v
```

## Sprint A.10: Train-Only Shuffled-Control Evidence Audit

The EEG v1 baseline checksum audit now verifies that the required `shuffled_target_control` row
states `shuffle_boundary=train_split_only`, train targets shuffled, and validation/test targets not
shuffled. A checksum-updated diagnostics artifact cannot relabel the negative control as all-split
shuffling while the gate still passes.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_non_train_only_shuffled_control -v
```

## Sprint A.11: Shuffled-Control Seed Provenance Audit

The EEG v1 baseline checksum audit now verifies that the required `shuffled_target_control` row
includes deterministic seed provenance for the train-only target shuffle. A checksum-updated
diagnostics artifact cannot drop `shuffle_seed` or `shuffle_seed_source` while the gate still passes.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_missing_shuffled_control_seed_provenance -v
```

## Sprint A.12: Shuffled-Control Seed Contract Audit

The EEG v1 baseline checksum audit now cross-checks the required `shuffled_target_control`
`shuffle_seed` against `run_config.seed + 1701` whenever the row declares
`shuffle_seed_source=diagnostic_seed_plus_1701`. A checksum-updated diagnostics artifact cannot
replace the deterministic train-shuffle seed while keeping the benchmark-readiness gate passing.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_shuffled_control_seed_run_config_mismatch -v
```

## Sprint A.13: Autocorrelation Summary MSE Integrity Audit

The EEG v1 baseline checksum audit now cross-checks the displayed autocorrelation summary MSE fields
against the completed diagnostic rows that generated them. A checksum-updated diagnostics artifact
cannot alter `short_horizon_best_mse`, `tiny_ssm_mse`, `shuffled_target_best_mse`, or
`shuffled_target_control_mse` while leaving the row-level evidence unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_autocorr_summary_mse_row_mismatch -v
```

## Sprint A.14: Autocorrelation Summary Delta Integrity Audit

The EEG v1 baseline checksum audit now recomputes the reviewer-facing autocorrelation delta fields
from completed diagnostic rows. A checksum-updated diagnostics artifact cannot alter
`long_horizon_delta_vs_short`, `non_overlap_delta_vs_short`, or `delta_prediction_delta_vs_short`
without changing the row-level MSE evidence that produced them.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_autocorr_summary_delta_row_mismatch -v
```

## Sprint A.15: Autocorrelation Summary Boolean Integrity Audit

The EEG v1 baseline checksum audit now recomputes the reviewer-facing autocorrelation boolean fields
from completed diagnostic rows. A checksum-updated diagnostics artifact cannot alter
`shuffled_control_degrades`, `persistence_or_ridge_dominates`, or
`shuffled_target_close_to_real_baselines` while leaving the row-level evidence unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_autocorr_summary_boolean_row_mismatch -v
```

## Sprint A.16: Model-Win Claim Recompute Audit

The EEG v1 baseline checksum audit now recomputes `model_win_claim_allowed`, `model_win_status`, and
`model_win_claim_failure_reasons` from the underlying metrics and autocorrelation diagnostics. A
checksum-updated bundle cannot edit both `metrics.json` and `evidence_gate.json` consistently to
claim a model win when ridge/persistence-style baselines still dominate.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_coordinated_false_model_win_claim -v
```

## Sprint A.17: Best-Baseline Summary Recompute Audit

The EEG v1 baseline checksum audit now recomputes `best_baseline`, `best_baseline_mse`,
`best_baseline_gap`, and `kahlus_beats_best_baseline` from `metrics_by_model`. A checksum-updated
bundle cannot alter derived baseline-win fields to make the model-win gate look earned while the
underlying model MSEs still show a baseline winning.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_best_baseline_summary -v
```

## Sprint A.18: Baseline Ranking Recompute Audit

The EEG v1 baseline checksum audit now recomputes `baseline_ranking` from `metrics_by_model`. A
checksum-updated bundle cannot reorder or rewrite reviewer-facing ranking rows while leaving the
underlying model MSE evidence unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_baseline_ranking -v
```

## Sprint A.19: Baseline Table JSON Recompute Audit

The EEG v1 baseline checksum audit now recomputes `baseline_table.json` rows and ranking from
`metrics_by_model`. A checksum-updated bundle cannot rewrite the reviewer-facing baseline table JSON
while leaving the underlying model metrics unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_baseline_table_json -v
```

## Sprint A.20: Baseline Table CSV Recompute Audit

The EEG v1 baseline checksum audit now recomputes `baseline_table.csv` rows from
`metrics_by_model`. A checksum-updated bundle cannot rewrite the reviewer-facing CSV table while
leaving the underlying model metrics unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_baseline_table_csv -v
```

## Sprint A.21: Metrics CSV Recompute Audit

The EEG v1 baseline checksum audit now recomputes `metrics.csv` rows from `metrics_by_model`.
A checksum-updated bundle cannot rewrite the reviewer-facing metric CSV while leaving the structured
metrics artifact unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_metrics_csv -v
```

## Sprint A.22: Granular Metrics CSV Recompute Audit

The EEG v1 baseline checksum audit now recomputes `per_subject_metrics.csv`,
`per_channel_metrics.csv`, and `per_horizon_metrics.csv` from the corresponding structured fields
in `metrics.json`. A checksum-updated bundle cannot rewrite reviewer-facing granular metric tables
while leaving the structured metrics artifact unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_granular_metric_csvs -v
```

## Sprint A.23: Autocorrelation Diagnostics CSV Recompute Audit

The EEG v1 baseline checksum audit now recomputes `autocorrelation_diagnostics.csv` from
`autocorrelation_diagnostics.json`. A checksum-updated bundle cannot rewrite the reviewer-facing
autocorrelation diagnostic table while leaving the structured diagnostic artifact unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_autocorr_diagnostics_csv -v
```

## Sprint A.24: Diagnostic Report Claim-Line Audit

The EEG v1 baseline checksum audit now checks `diagnostic_report.md` against `metrics.json` and
`evidence_gate.json` for the report's visible claim scope, scientific-claim status, best baseline,
and Kahlus-vs-best-baseline result. A checksum-updated bundle cannot rewrite those human-facing
report lines while leaving the structured evidence unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_diagnostic_report_claim_lines -v
```

## Sprint A.25: Target Scale Context Recompute Audit

The EEG v1 baseline checksum audit now recomputes each model's `target_scale_context.json` RMSE,
RMSE-relative-to-target-std, and MSE-relative-to-target-variance fields from `metrics.json` plus the
recorded target scale. A checksum-updated bundle cannot make normalized EEG MSE look smaller relative
to target scale while leaving the structured metrics unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_target_scale_context -v
```

## Sprint A.26: Dataset Summary Consistency Audit

The EEG v1 baseline checksum audit now checks `dataset_summary.json` against `metrics.json`,
`split_audit.json`, and `run_config.json` for dataset/source/status, split type, subject counts, and
run window settings. A checksum-updated bundle cannot relabel a synthetic fixture as a public
benchmark or alter held-out split context while leaving the structured evidence unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_dataset_summary -v
```

## Sprint A.27: Run Config Consistency Audit

The EEG v1 baseline checksum audit now checks `run_config.json` against `metrics.json`,
`evidence_gate.json`, and `dataset_summary.json` for dataset/source/status, claim scope, configured
model coverage, selection policy, and run window settings. A checksum-updated bundle cannot hide a
required completed baseline, broaden the claim scope, or relabel the run configuration while leaving
the structured evidence unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_run_config -v
```

## Sprint A.28: Split Audit Internal Consistency

The EEG v1 baseline checksum audit now recomputes subject overlap from `split_audit.json` subject
lists and checks that `subject_overlap` and `leakage_passed` agree with the listed subjects and
failure reasons. A checksum-updated bundle cannot hide train/validation/test subject overlap inside
the split audit artifact while keeping the same subject counts.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_split_audit_subject_overlap -v
```

## Sprint A.29: Target Scale Variance Consistency Audit

The EEG v1 baseline checksum audit now checks that `target_scale_context.json` keeps
`target_variance` consistent with `target_std ** 2`. A checksum-updated bundle cannot distort the
normalized-MSE scale denominator while also rewriting the dependent per-model variance ratios.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_target_scale_variance_std_mismatch -v
```

## Sprint A.30: Evidence Gate Finite-Metrics Consistency Audit

The EEG v1 baseline checksum audit now recomputes the `finite_metrics` gate field from
`metrics.json`. A checksum-updated bundle cannot leave `evidence_gate.json` claiming a clean
benchmark-readiness gate while disagreeing with the actual metric finiteness payload.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_gate_finite_metrics_mismatch -v
```

## Sprint A.31: Evidence Gate Baseline-Table Consistency Audit

The EEG v1 baseline checksum audit now recomputes the `baseline_table_present` gate field from
`metrics.json`. A checksum-updated bundle cannot make the gate disagree with whether completed
baseline results exist.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_gate_baseline_table_present_mismatch -v
```

## Sprint A.32: Evidence Gate Split-Audit Consistency Audit

The EEG v1 baseline checksum audit now checks `evidence_gate.json["split_audit_passed"]` against
`split_audit.json["leakage_passed"]`. A checksum-updated bundle cannot make the gate disagree with
the subject-held-out leakage audit result.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_gate_split_audit_passed_mismatch -v
```

## Sprint A.33: Evidence Gate Dataset Consistency Audit

The EEG v1 baseline checksum audit now checks `evidence_gate.json["dataset"]` against
`metrics.json["dataset"]`. A checksum-updated bundle cannot relabel a synthetic fixture gate as a
different dataset while leaving the structured metrics unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_gate_dataset_mismatch -v
```

## Sprint A.34: Evidence Gate Branch Consistency Audit

The EEG v1 baseline checksum audit now checks `evidence_gate.json["branch"] == "v1"`. A
checksum-updated bundle cannot relabel a v1 EEG benchmark-readiness gate as another branch.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_gate_branch_mismatch -v
```

## Sprint A.35: Evidence Gate Claim-Scope Consistency Audit

The EEG v1 baseline checksum audit now checks `evidence_gate.json["claim_scope"]` against
`eeg_future_forecasting_benchmark_ready`. A checksum-updated bundle cannot relabel the v1 EEG
benchmark-readiness gate as diagnosis, treatment, or another broader claim scope.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_gate_claim_scope_mismatch -v
```

## Sprint A.36: Evidence Gate Allowed-Claim-Scope Consistency Audit

The EEG v1 baseline checksum audit now checks
`evidence_gate.json["gate_criteria"]["allowed_claim_scope"]` against
`eeg_future_forecasting_benchmark_ready`. A checksum-updated bundle cannot keep the top-level v1
claim narrow while broadening the nested gate criteria shown to reviewers.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_gate_allowed_claim_scope_mismatch -v
```

## Sprint A.37: Evidence Gate Required-Control Criteria Consistency Audit

The EEG v1 baseline checksum audit now checks
`evidence_gate.json["gate_criteria"]["required_first_class_baselines"]` and
`evidence_gate.json["gate_criteria"]["required_negative_controls"]` against the required TinySSM
baseline and shuffled-target control. A checksum-updated bundle cannot hide the baseline/control
requirements from the reviewer-facing gate criteria.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_gate_required_control_criteria_mismatch -v
```

## Sprint A.38: Evidence Gate Shuffled-Control Criteria Consistency Audit

The EEG v1 baseline checksum audit now checks
`evidence_gate.json["gate_criteria"]["requires_shuffled_target_degradation"]` and
`evidence_gate.json["gate_criteria"]["requires_shuffled_target_not_close_to_real_baselines"]`
against the required shuffled-target safeguards. A checksum-updated bundle cannot soften the
negative-control criteria while keeping the report internally consistent.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_gate_shuffled_control_criteria_mismatch -v
```

## Sprint A.39: Evidence Gate Core Criteria Consistency Audit

The EEG v1 baseline checksum audit now checks the core reviewer-facing gate criteria:
minimum forecast horizon, allowed split types, split-audit requirement, baseline-table
requirement, finite-metrics requirement, and calibration requirement. A checksum-updated bundle
cannot soften these criteria while keeping the report internally consistent.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_gate_core_criteria_mismatch -v
```

## Sprint A.40: Diagnostic Report Gate-Criteria Consistency Audit

The EEG v1 baseline checksum audit now checks the gate-criteria lines in `diagnostic_report.md`
against `evidence_gate.json`. A checksum-updated bundle cannot keep structured gate JSON intact
while showing softened criteria to reviewers in the Markdown report.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_diagnostic_report_gate_criteria -v
```

## Sprint A.41: Diagnostic Report Model-Win Consistency Audit

The EEG v1 baseline checksum audit now checks the model-win lines and model-win failure reasons
in `diagnostic_report.md` against `evidence_gate.json`. A checksum-updated bundle cannot keep the
structured gate blocked while making the Markdown report claim that Kahlus beat the baselines.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_diagnostic_report_model_win -v
```

## Sprint A.42: Diagnostic Report Target-Scale Consistency Audit

The EEG v1 baseline checksum audit now checks the target-scale lines in `diagnostic_report.md`
against `target_scale_context.json`. A checksum-updated bundle cannot keep the structured target
scale intact while showing misleading normalized-MSE scale context in the Markdown report.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_diagnostic_report_target_scale -v
```

## Sprint A.43: Diagnostic Report Target-Units Consistency Audit

The EEG v1 baseline checksum audit coverage now explicitly tampers `target_units` in
`diagnostic_report.md` and requires the audit to reject the bundle against
`target_scale_context.json`. A checksum-updated bundle cannot make normalized fixture MSE look like
raw microvolt-scale evidence in the Markdown report.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_diagnostic_report_target_scale -v
```

## Sprint A.44: Diagnostic Report Autocorrelation Summary Consistency Audit

The EEG v1 baseline checksum audit now checks reviewer-facing autocorrelation summary lines in
`diagnostic_report.md` against `autocorrelation_diagnostics.json`. A checksum-updated bundle cannot
hide ridge/persistence dominance or make shuffled-target controls look safer in the Markdown report.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_diagnostic_report_autocorr_summary -v
```

## Sprint A.45: Diagnostic Report Autocorrelation Verdict Consistency Audit

The EEG v1 baseline checksum audit now also checks the autocorrelation control deltas,
`shuffled_control_degrades`, and verdict lines in `diagnostic_report.md` against
`autocorrelation_diagnostics.json`. A checksum-updated bundle cannot preserve the structured
diagnostics while softening the reviewer-facing autocorrelation verdict.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_diagnostic_report_autocorr_summary -v
```

## Sprint A.46: Diagnostic Report Autocorrelation Row Consistency Audit

The EEG v1 baseline checksum audit now checks each autocorrelation diagnostic table row in
`diagnostic_report.md` against `autocorrelation_diagnostics.json`. A checksum-updated bundle cannot
keep the structured shuffled-target control evidence intact while softening the Markdown row that
reviewers read.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_diagnostic_report_autocorr_row -v
```

## Sprint A.47: Diagnostic Report Artifact-Index Consistency Audit

The EEG v1 baseline checksum audit now checks the artifact-index rows in `diagnostic_report.md`.
A checksum-updated bundle cannot keep autocorrelation diagnostics in the machine artifacts while
hiding them from the reviewer-facing artifact list.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_diagnostic_report_artifact_index -v
```

## Sprint A.48: Diagnostic Report Verification-Lane Consistency Audit

The EEG v1 baseline report now prints the local-only execution lane and `a100_jobs_launched` value
from `baseline_verification.json`, and the checksum audit rejects Markdown-only tampering. A
checksum-updated bundle cannot make the reviewer-facing report imply that A100 work was launched.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_diagnostic_report_verification_lines -v
```

## Sprint A.49: Diagnostic Report Method-Order Consistency Audit

The EEG v1 baseline checksum audit now checks the `Method Order` rows in `diagnostic_report.md`
against `run_config.json`. A checksum-updated bundle cannot make the reviewer-facing report show
the main model before baselines while the replay config remains baseline-first.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_diagnostic_report_method_order -v
```

## Sprint A.50: Diagnostic Report Baseline-Ranking Consistency Audit

The EEG v1 baseline checksum audit now checks the `Baseline Ranking` rows in `diagnostic_report.md`
against `metrics.json`. A checksum-updated bundle cannot make the reviewer-facing ranking claim a
different best model while the structured baseline ranking remains unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_diagnostic_report_baseline_ranking -v
```

## Sprint A.51: Diagnostic Report Baseline-Gap Consistency Audit

The EEG v1 baseline checksum audit now checks the `Baseline Gap Summary` lines in
`diagnostic_report.md` against `metrics.json`. A checksum-updated bundle cannot soften the
reviewer-facing persistence/ridge/best-baseline gap while keeping structured metrics unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_diagnostic_report_baseline_gaps -v
```

## Sprint A.52: Diagnostic Report Duplicate Protected-Line Audit

The EEG v1 baseline checksum audit now rejects duplicate protected lines in `diagnostic_report.md`.
A checksum-updated bundle cannot keep the correct structured line while appending a second
reviewer-facing claim or baseline line that creates ambiguity.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_duplicate_diagnostic_report_claim_lines -v
```

## Sprint A.53: Diagnostic Report Model-Win Reason List Audit

The EEG v1 baseline checksum audit now checks the rendered `model_win_claim_failure_reasons`
bullet list in `diagnostic_report.md` exactly against `evidence_gate.json`. A checksum-updated
bundle cannot append extra reviewer-facing model-win reasoning while keeping the gate unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_extra_diagnostic_report_model_win_reason -v
```

## Sprint A.54: Diagnostic Report Target-Scale Model Row Audit

The EEG v1 baseline checksum audit now checks each target-scale model row in `diagnostic_report.md`
against `target_scale_context.json`. A checksum-updated bundle cannot make normalized-MSE scale
context look better in the reviewer-facing report while keeping the structured artifact unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_diagnostic_report_target_scale_model_row -v
```

## Sprint A.55: Diagnostic Report Target-Scale Row-List Audit

The EEG v1 baseline checksum audit now checks the full target-scale model-row list in
`diagnostic_report.md` against `target_scale_context.json`. A checksum-updated bundle cannot append
extra reviewer-facing target-scale rows while keeping the structured artifact unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_extra_diagnostic_report_target_scale_model_row -v
```

## Sprint A.56: Diagnostic Report Metric-Breakdown Count Audit

The EEG v1 baseline checksum audit now checks `Metric Breakdown Summary` row counts in
`diagnostic_report.md` against `metrics.json`. A checksum-updated bundle cannot hide or inflate the
reviewer-facing per-subject/channel/horizon sidecar coverage counts while keeping structured
metrics unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_diagnostic_report_metric_breakdown_counts -v
```

## Sprint A.57: Diagnostic Report Metric Sidecar List Audit

The EEG v1 baseline checksum audit now checks the `detailed_sidecars` line in
`diagnostic_report.md`. A checksum-updated bundle cannot point reviewers away from the
per-subject/channel/horizon metric sidecars while keeping those artifacts unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_diagnostic_report_metric_sidecars -v
```

## Sprint A.58: Diagnostic Report Run-Config Line Audit

The EEG v1 baseline checksum audit now checks the `Run Config` section in
`diagnostic_report.md` against `run_config.json`. A checksum-updated bundle cannot change
reviewer-facing seed, model list, windowing, data-source, selection-policy, or claim-scope lines
while keeping the structured run config unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_diagnostic_report_run_config_lines -v
```

## Sprint A.59: Diagnostic Report Method-Order Row-List Audit

The EEG v1 baseline checksum audit now checks the full `Method Order` table in
`diagnostic_report.md` against `run_config.json`. A checksum-updated bundle cannot append extra
reviewer-facing model rows while keeping the configured model list unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_extra_diagnostic_report_method_order_row -v
```

## Sprint A.60: Diagnostic Report Baseline-Ranking Row-List Audit

The EEG v1 baseline checksum audit now checks the full `Baseline Ranking` table in
`diagnostic_report.md` against `metrics.json`. A checksum-updated bundle cannot append extra
reviewer-facing ranking rows while keeping the structured metrics unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_extra_diagnostic_report_baseline_ranking_row -v
```

## Sprint A.61: Diagnostic Report Artifact-Index Row-List Audit

The EEG v1 baseline checksum audit now checks the full `Artifact Index` table in
`diagnostic_report.md` against the canonical baseline artifact list. A checksum-updated bundle
cannot append extra reviewer-facing artifact rows while keeping the actual evidence manifest
unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_extra_diagnostic_report_artifact_index_row -v
```

## Sprint A.62: Diagnostic Report Autocorrelation Row-List Audit

The EEG v1 baseline checksum audit now checks the full autocorrelation diagnostics table in
`diagnostic_report.md` against `autocorrelation_diagnostics.json`. A checksum-updated bundle
cannot append extra reviewer-facing autocorrelation/control rows while keeping the structured
diagnostics unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_extra_diagnostic_report_autocorr_row -v
```

## Sprint A.63: Diagnostic Report Metric-Breakdown Section Audit

The EEG v1 baseline checksum audit now checks the full `Metric Breakdown Summary` section in
`diagnostic_report.md` against `metrics.json`. A checksum-updated bundle cannot append extra
reviewer-facing sidecar/count lines while keeping the structured metrics unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_extra_diagnostic_report_metric_breakdown_line -v
```

## Sprint A.64: Diagnostic Report Gate-Criteria Section Audit

The EEG v1 baseline checksum audit now checks the full `Evidence Gate Criteria` section in
`diagnostic_report.md` against `evidence_gate.json`. A checksum-updated bundle cannot append extra
reviewer-facing gate criteria while keeping the structured gate unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_extra_diagnostic_report_gate_criteria_line -v
```

## Sprint A.65: Diagnostic Report Target-Scale Header Audit

The EEG v1 baseline checksum audit now checks the target-scale header bullets in
`diagnostic_report.md` against `target_scale_context.json`, including the scale note. A
checksum-updated bundle cannot append extra reviewer-facing target-scale interpretation lines while
keeping the structured scale context unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_extra_diagnostic_report_target_scale_header_line -v
```

## Sprint A.66: Diagnostic Report Checksum-Audit Section Audit

The EEG v1 baseline checksum audit now checks the full `Checksum Audit` section in
`diagnostic_report.md` against `baseline_verification.json` and the canonical local audit command.
A checksum-updated bundle cannot append extra reviewer-facing cluster-readiness or verification
claims while keeping the structured verification sidecar unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_extra_diagnostic_report_checksum_audit_line -v
```

## Sprint A.67: Diagnostic Report Claim-Boundaries Section Audit

The EEG v1 baseline checksum audit now checks the full `Claim Boundaries` section in
`diagnostic_report.md` against the narrow v1 benchmark-readiness claim boundary. A
checksum-updated bundle cannot append reviewer-facing clinical, SOTA, foundation-model, v2, or v3
claims while keeping the structured evidence artifacts unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_extra_diagnostic_report_claim_boundary_line -v
```

## Sprint A.68: Diagnostic Report Summary-Header Section Audit

The EEG v1 baseline checksum audit now checks the top summary block in `diagnostic_report.md`
against `metrics.json`, `evidence_gate.json`, and `split_audit.json`. A checksum-updated bundle
cannot append extra reviewer-facing model-win or claim-scope lines before the artifact index while
keeping the structured evidence artifacts unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_extra_diagnostic_report_summary_header_line -v
```

## Sprint A.69: Diagnostic Report Baseline-Gap Section Audit

The EEG v1 baseline checksum audit now checks the full `Baseline Gap Summary` section in
`diagnostic_report.md` against `metrics.json`. A checksum-updated bundle cannot append extra
reviewer-facing calibrated-gap or model-win lines while keeping the structured baseline gaps
unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_extra_diagnostic_report_baseline_gap_line -v
```

## Sprint A.70: Diagnostic Report Autocorrelation-Summary Row Audit

The EEG v1 baseline checksum audit now checks the full autocorrelation `Summary` table in
`diagnostic_report.md` against `autocorrelation_diagnostics.json`. A checksum-updated bundle cannot
append extra reviewer-facing autocorrelation-cleared rows while keeping the structured diagnostics
unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_extra_diagnostic_report_autocorr_summary_row -v
```

## Sprint A.71: Diagnostic Report Autocorrelation-Dominance Bullet Audit

The EEG v1 baseline checksum audit now checks the `Baseline Dominance` bullets in
`diagnostic_report.md` against `autocorrelation_diagnostics.json`. A checksum-updated bundle cannot
append extra reviewer-facing autocorrelation-cleared dominance lines while keeping the structured
diagnostics unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_extra_diagnostic_report_autocorr_dominance_line -v
```

## Sprint A.72: Diagnostic Report Autocorrelation Intro Audit

The EEG v1 baseline checksum audit now checks the opening autocorrelation warning and caveat in
`diagnostic_report.md` against `autocorrelation_diagnostics.json` plus the fixed claim-hygiene
caveat. A checksum-updated bundle cannot append reviewer-facing autocorrelation-cleared prose before
the summary table while keeping the structured diagnostics unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_extra_diagnostic_report_autocorr_intro_line -v
```

## Sprint A.73: Diagnostic Report Model-Win Section Audit

The EEG v1 baseline checksum audit now checks the full `Model Win Claim Status` section in
`diagnostic_report.md` against `evidence_gate.json`. A checksum-updated bundle cannot append extra
reviewer-facing calibrated model-win override lines while keeping the structured gate unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_extra_diagnostic_report_model_win_line -v
```

## Sprint A.74: Failure-Reasons Sidecar Consistency Audit

The EEG v1 baseline checksum audit now checks `failure_reasons.json` baseline and diagnostic failure
rows against the authoritative metrics and autocorrelation artifacts. A checksum-updated bundle
cannot hide baseline runner failures or missing-control diagnostic failures while leaving the source
artifacts unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_failure_reasons_baseline_and_diagnostic_mismatch -v
```

## Sprint A.75: Autocorrelation Row Metric Consistency Audit

The EEG v1 baseline checksum audit now checks each completed autocorrelation diagnostic row so its
published `persistence_mse`, `linear_ridge_mse`, `tiny_ssm_mse`, `best_model`, and `best_mse` fields
match the nested `metrics_by_model` values. A checksum-updated bundle cannot make TinySSM or the
winning autocorrelation baseline look different in summary fields than in the structured row metrics.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_autocorr_row_metric_field_mismatch -v
```

## Sprint A.76: Diagnostic Report Failure-Section Audit

The EEG v1 baseline checksum audit now checks the `Gate Failures`, `Split Audit Failures`, and
`Baseline Failures` sections in `diagnostic_report.md` against the structured gate, split-audit, and
metrics failure payloads. A checksum-updated bundle cannot append reviewer-facing failure lines that
do not exist in the source artifacts.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_extra_diagnostic_report_failure_lines -v
```

## Sprint A.77: Diagnostic Report Stimulus/Task Split Audit

The EEG v1 baseline checksum audit now checks the `Stimulus/Task Split Audit` section in
`diagnostic_report.md` against the structured `stimulus_task_held_out_split` diagnostic row. A
checksum-updated bundle cannot add reviewer-facing stimulus/task leakage overrides while keeping the
structured diagnostic unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_extra_diagnostic_report_stimulus_task_audit_line -v
```

## Sprint A.78: HBN Local Boundary Report Audit

The EEG v1 baseline checksum audit now checks the `HBN Local Path Boundary` section in
`diagnostic_report.md` whenever `benchmark_status` is `local_manifest_not_public_hbn_benchmark`.
A checksum-updated bundle cannot turn local HBN-style manifest evidence into reviewer-facing public
HBN benchmark evidence.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_hbn_local_boundary -v
```

## Sprint A.79: Dataset Summary Granular Count Audit

The EEG v1 baseline checksum audit now checks `dataset_summary.json` test-window and channel counts
against `per_subject_metrics.csv` and `per_channel_metrics.csv`. A checksum-updated bundle cannot
inflate reviewer-facing test-window or channel counts while leaving granular metric sidecars unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_dataset_summary_granular_count_mismatch -v
```

## Sprint A.80: Diagnostic Report Dataset Summary Audit

The EEG v1 diagnostic report now includes a bounded `Dataset Summary` section sourced from
`dataset_summary.json`, and the checksum audit checks that section exactly. A checksum-updated bundle
cannot inflate reviewer-facing subject, window, channel, or window-geometry counts in Markdown while
leaving the structured dataset summary unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_a.EEGV1SprintATests.test_baseline_checksum_audit_rejects_tampered_diagnostic_report_dataset_summary -v
```
