# Kahlus v1 Few-Shot Adaptation Plan

## Scope

Sprint B compares local few-shot subject adaptation strategies on the Kahlus v1 EEG
future-window fixture after Sprint A baselines pass. It is benchmark-readiness work only.

## Methods

Baselines come first:

```text
support_persistence
support_ridge
```

Adaptation methods:

```text
linear_probe
bottleneck_adapter
full_finetune
```

`bottleneck_adapter` is a local lightweight adapter baseline, not a LoRA claim. A real LoRA
integration requires a pinned model surface and a separate comparison plan.

## Claim Boundary

Allowed scope:

```text
eeg_fewshot_adaptation_benchmark_ready
```

Blocked: adapter superiority, diagnosis, treatment, clinical use, depression or epilepsy
detection, foundation-model claims, SOTA, v2/v3 success, and recovery claims.

## A100 Boundary

This sprint must not launch A100 or cluster jobs. The 7x NVIDIA A100 80GB cluster is a later
scaling asset only after local synthetic fixtures, subject-held-out split audit, local baseline
ladder, and evidence gates pass from a clean merge/tag.

Future A100 packaging must label the system honestly as `7xA100`, include clean-worktree proof,
exact commit hash, runner tarball, configs, checksum manifest, CPU smoke test, DDP/torchrun
command, evidence bundle writer, audit script, and exclude secrets, checkpoints unless explicitly
allowed, and raw participant data.

## Verification

```bash
PYTHONPATH=src python3 scripts/run_eeg_v1_adaptation.py --dataset synthetic_fixture --out-dir /tmp/kahlus_v1_adapt_smoke --seed 0 --pretrain-steps 2 --adapt-steps 2 --support-windows 4
```

## Sprint C: Invalid Task Configuration CLI Guard

The few-shot adaptation runner now reports impossible support/query window settings as a clear
local configuration error instead of a Python traceback. Invalid adaptation geometry is not an
adaptation result and must fail before evidence artifacts or benchmark-readiness claims are
written.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint D: Adaptation Dataset Summary Evidence Artifact

The few-shot adaptation runner now writes `adaptation_dataset_summary.json` with bounded,
non-raw metadata: dataset ID, pretrain/support/query window counts, adaptation subject count,
window settings, channel count, and method count. This gives reviewers an adaptation geometry
check without exposing raw EEG or expanding the benchmark-readiness claim.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint E: Adaptation Report Dataset Summary

The few-shot adaptation Markdown report now includes the same bounded dataset-summary fields
written to `adaptation_dataset_summary.json`. This keeps reviewer-facing evidence self-contained
while preserving the no-raw-EEG and benchmark-readiness-only boundaries.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint F: Adaptation Split Audit Artifact

The few-shot adaptation runner now writes `adaptation_split_audit.json` and derives the
adaptation evidence gate from that same split-audit payload. This preserves the subject-held-out
leakage evidence alongside adaptation metrics instead of reducing it to an internal boolean.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint G: Adaptation Report Split Audit Summary

The few-shot adaptation Markdown report now includes a bounded split-audit summary with split
type, leakage status, overlap flags, and failure-reason count. This makes the reviewer-facing
report self-contained for subject-held-out evidence without exposing raw EEG or participant data.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint H: Adaptation Local Provenance Boundary

The few-shot adaptation runner now carries `data_source` and `benchmark_status` through stdout,
metrics, run config, dataset summary, and the Markdown report. HBN-style local manifest runs are
explicitly marked `local_manifest_not_public_hbn_benchmark` so fixture smoke results cannot be
presented as public HBN benchmark evidence.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint I: Adaptation Report Baseline-First Method Order

The few-shot adaptation Markdown report now includes a `Method Order` table before the performance
ranking. Support baselines are listed first and tagged as baselines, so ranking output cannot
obscure the rule that baselines are results and come before adaptation methods.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint J: Adaptation Failure Reasons Artifact

The few-shot adaptation runner now writes `adaptation_failure_reasons.json` with gate failures
and split-audit failures. This gives audit consumers one bounded place to inspect why an
adaptation evidence gate would block a benchmark-readiness claim.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint K: Adaptation Run Config Reproducibility Fields

The few-shot adaptation run config now records the seed, pretrain step budget, adaptation step
budget, and query-window count alongside the existing model/method settings. This keeps the
replay-critical knobs in `adaptation_run_config.json`, not only in metrics output.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint L: Adaptation Report Run Config Summary

The few-shot adaptation Markdown report now includes a `Run Config` section with the seed,
training/adaptation step budgets, window length, forecast horizon, support-window count, and
query-window count. This makes the human-readable evidence report replayable without opening
the JSON sidecar first.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint M: Adaptation Evidence Gate Criteria

The few-shot adaptation evidence gate now records its decision criteria directly in
`adaptation_evidence_gate.json`: minimum support/query windows, required split audit,
baseline table, finite metrics, calibration check, and the allowed narrow claim scope.
The Markdown report renders the same criteria so reviewers can see that the gate is a
benchmark-readiness gate, not an adapter-superiority or clinical threshold.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint N: Adaptation Report Gate Failure Summary

The few-shot adaptation Markdown report now includes an `Evidence Gate Failures` section when
the gate blocks the benchmark-readiness claim. This keeps blocked local evidence bundles
reviewable from the report itself while preserving the separate structured
`adaptation_failure_reasons.json` sidecar.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint O: Adaptation Report Split-Audit Failure Summary

The few-shot adaptation Markdown report now includes a `Split Audit Failures` section when the
split audit records detailed leakage or split-validation failures. This surfaces the underlying
split evidence in the human-readable report instead of only showing the derived gate failure.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint P: Adaptation Sampling Rate Provenance

The few-shot adaptation task now carries a single finite dataset sampling rate into
`adaptation_run_config.json`, `adaptation_dataset_summary.json`, and the adaptation Markdown
report. This keeps local HBN-style adaptation windows auditable in seconds/Hz terms without
changing model training or claiming public HBN benchmark status.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint Q: Adaptation Target Scale Context

The few-shot adaptation evidence bundle now writes `adaptation_target_scale_context.json` and
renders the same target-scale summary in the Markdown report. The context reports held-out query
target distribution statistics plus per-method RMSE/MSE ratios against target scale, so low
adaptation MSE is interpreted in normalized fixture units rather than raw EEG microvolts or
adapter understanding.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint R: Adaptation Target Scale Context in Smoke Stdout

The few-shot adaptation runner now prints target units, target standard deviation, target
variance, and the best method's RMSE relative to target standard deviation directly in stdout.
This keeps local adaptation smoke results interpretable without opening the JSON sidecar, while
preserving the benchmark-readiness-only claim boundary.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint S: Adaptation Baseline Gap Summary

The few-shot adaptation evidence bundle now writes `adaptation_baseline_gap_summary.json` and
renders the same summary in the Markdown report. It compares the best adaptation method against
the best support baseline, preserving the rule that support baselines are results and that
adapter-win discussion requires beating the best support baseline first.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint T: Adaptation Baseline Gap Summary in Smoke Stdout

The few-shot adaptation runner now prints the best support baseline, best adaptation method,
their MSE delta, and whether the best adaptation method beats the best support baseline directly
in stdout. This keeps baseline-first interpretation visible in the one-line smoke output without
claiming adapter superiority.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint U: Adaptation Metric Breakdown Summary in Report

The few-shot adaptation Markdown report now summarizes how many subject-level metric rows were
written and points to `adaptation_subject_metrics.csv`. This keeps reviewer-facing evidence
auditable without inlining every per-subject row in the report.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint V: Adaptation Metric Breakdown in Smoke Stdout

The few-shot adaptation runner now prints the subject-level metric row count and
`adaptation_subject_metrics.csv` artifact path directly to stdout. This keeps the one-command
local smoke output tied to the subject-level sidecar without changing adaptation training or
claim scope.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint W: Per-Subject Adaptation Baseline Gap Summary

The few-shot adaptation evidence bundle now writes
`adaptation_subject_baseline_gap_summary.json` and renders the same bounded summary in the
Markdown report. Each held-out subject is compared against that subject's best support baseline
before any adapter-win interpretation is allowed.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint X: Per-Subject Baseline Gap in Smoke Stdout

The few-shot adaptation runner now prints the count of held-out subjects where adaptation beats
the best support baseline and the `adaptation_subject_baseline_gap_summary.json` artifact path.
This keeps subject-level adapter-win evidence visible in the one-command local smoke output.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint Y: Adaptation Report Artifact Index

The few-shot adaptation Markdown report now includes an `Artifact Index` table listing the bounded
JSON/CSV sidecars, including target-scale, baseline-gap, per-subject metric, and per-subject
baseline-gap artifacts. This keeps evidence review navigable without changing adaptation training,
gates, or claim scope.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint Z: Adaptation Evidence Checksum Manifest

The few-shot adaptation evidence bundle now writes `adaptation_checksum_manifest.json` with SHA-256
digests and byte counts for the emitted JSON/CSV/Markdown artifacts. The manifest excludes itself
to avoid circular hashing and gives reviewers a local integrity check before any later handoff.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint AA: Adaptation Checksum Audit Script

The few-shot adaptation lane now includes `scripts/audit_eeg_v1_adaptation_checksums.py`, a local
JSON-emitting verifier for `adaptation_checksum_manifest.json`. It fails closed on missing
artifacts, byte-count changes, checksum mismatches, unsupported schemas, and unsupported algorithms,
so the checksum manifest becomes a runnable evidence gate.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint AB: Adaptation Checksum Manifest in Smoke Stdout

The few-shot adaptation runner now prints the `adaptation_checksum_manifest.json` path in stdout
next to the subject-level evidence artifacts and gate path. This keeps the one-command smoke output
connected to the checksum audit input without changing training, ranking, gates, or claim scope.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint AC: Adaptation Report Checksum Audit Instructions

The few-shot adaptation Markdown report now includes a `Checksum Audit` section that names
`adaptation_checksum_manifest.json` and gives the local audit command using an explicit
`<artifact-dir>` placeholder. This keeps the report self-contained for evidence review without
changing adaptation training, ranking, gates, or claim scope.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint AD: Concrete Checksum Audit Command in Smoke Stdout

The few-shot adaptation runner now prints a concrete `checksum_audit_command` line using the
actual `--out-dir` value from the smoke run. This makes the checksum gate copy/pasteable from
stdout while preserving the local-only lane and leaving training, ranking, gates, and claim scope
unchanged.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint AE: Adaptation Verification Sidecar

The few-shot adaptation artifact bundle now writes `adaptation_verification.json`, a
machine-readable sidecar that records the local-only execution lane, `a100_jobs_launched=false`,
the checksum manifest name, and the exact checksum audit command for the emitted artifact
directory. The checksum manifest covers this sidecar, so a bundle audit can verify the
verification instructions as evidence rather than relying on stdout or Markdown prose alone.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint AF: Verification Sidecar Contract Audit

The checksum audit now validates `adaptation_verification.json` semantically in addition to
checking its SHA-256 digest. The audit fails if the sidecar no longer declares the local-only
execution lane, `a100_jobs_launched=false`, the expected checksum manifest, or the exact checksum
audit command for the artifact directory.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint AG: Verification Sidecar Manifest Coverage

The checksum audit now fails if `adaptation_verification.json` exists but is missing from
`adaptation_checksum_manifest.json`. This makes the prior verification-sidecar claim auditable:
the sidecar must be both semantically valid and covered by the SHA-256 manifest.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint AH: Duplicate Checksum Manifest Entry Rejection

The checksum audit now rejects duplicate artifact paths inside `adaptation_checksum_manifest.json`.
This prevents an evidence bundle from presenting ambiguous repeated checksum rows for the same
artifact while preserving the existing byte and SHA-256 validation.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint AI: Required Artifact Checksum Manifest Coverage

The checksum audit now requires `adaptation_checksum_manifest.json` to include every emitted
adaptation evidence artifact, including metrics, tables, summaries, report, gate, split audit,
failure reasons, and verification sidecar. Removing any required artifact row now fails the audit
even when the file still exists in the artifact directory.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint AJ: Unexpected Checksum Manifest Entry Rejection

The checksum audit now rejects validly check-summed but unexpected entries in
`adaptation_checksum_manifest.json`. The manifest is therefore bounded to the declared adaptation
evidence artifacts and cannot silently bless extra files outside the local adaptation evidence
bundle.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```

## Sprint AK: Adaptation Verification Sidecar in Smoke Stdout

The few-shot adaptation runner now prints the `adaptation_verification.json` path directly in smoke
stdout. This keeps the one-command local adaptation smoke output connected to the checksum-covered
verification sidecar without changing adaptation training, ranking, gates, or claim scope.

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.eeg_v1.test_eeg_v1_sprint_b_adaptation -v
```
