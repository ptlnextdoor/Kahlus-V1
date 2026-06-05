# MOABB Paper-Audit Evidence v1

Status: frozen reproducibility evidence baseline

Commit: `e9de1d20ebac406fe220673e7273f97e8ba5d48f`

## Claim Scope

This evidence bundle supports the Track A reproducibility/model-gates story:
NeuroTwin can run leakage-audited MOABB EEG evaluation, produce paper-mode
artifacts, surface subject identity risk, and keep unsupported scientific claims
disabled.

It does not support model-superiority, clinical, diagnostic, SOTA, or first
foundation-model claims.

## What Passed

- Real MOABB BNCI2014_001 prepared data path.
- Subject-held-out split audit.
- Paper-mode gate with required seeds `0,1,2`.
- Baseline ranking artifacts.
- Identity probe showing elevated subject recoverability.
- Full 6x A100 DDP training completed stably.

## What Did Not Pass

The run does not prove NeuroTwin beats simple baselines on MOABB. Linear ridge
beats the full model on the reported forecasting and masked-reconstruction
metrics, so this bundle must remain infrastructure/reproducibility evidence.

## Bundle Limitation

The old evidence zip is reviewable but not recomputable offline. It intentionally
excludes prepared `.npz` event arrays, checkpoints, raw data, and large runtime
artifacts. The included manifests, metrics, model card, leakage diagnostics, and
checksums are enough for audit/report review, but rerunning diagnostics requires
the prepared event arrays on the original prepared-data root or a regenerated
MOABB preparation.

## Next Track A Step

Run the Brookshire-style MOABB motor-imagery classification leakage demo on real
prepared MOABB manifests and bundle the resulting artifacts. The demo keeps
model, preprocessing, and task constant while comparing:

- `bad_segment_split`: random window split that leaks subject/window identity.
- `correct_subject_heldout`: subject-held-out generalization candidate.

The bad split remains a negative control and is never claim eligible.
