# Repository-Wide Bug Audit

## P0 Findings

### P0-001: Context and target overlap invalidates the central task wording

`src/neurotwin/data/prepared_tasks.py:268-272` creates `x=signal[:-1]` and `y=signal[1:]`. For a 128-sample prepared window, both arrays have length 127 and share 126 samples. `src/neurotwin/eeg_v1/dataset.py:184-202` similarly starts the target at `start + forecast_horizon`; the default horizon of one shares all but one sample with the input window. Stride can separate examples but cannot remove context-target overlap within an example.

Impact: the recovered 3.116 MSE does not validate distinct future-window forecasting. Existing tests encode and normalize this definition rather than detecting it.

### P0-002: Architecture attribution mismatch

The recovered result uses `NeuralStateSpaceTranslator` with `ssm_fallback`. `src/neurotwin/models/torch_models.py:384-397` shows that fallback is a GRU. The result is not an evaluation of `NeuralFieldCompiler` and not an SSM/Mamba result.

### P0-003: Inferential unit is wrong in baseline intervals

`src/neurotwin/benchmarks/baseline_suite.py` bootstraps flattened elementwise squared errors. Samples from the same subject, recording, time neighborhood, and channel are correlated. Element-level intervals substantially overstate effective sample size.

### P0-004: Baseline comparison is not budget-matched

The trained model uses a recovered 100,000-step checkpoint. Neural baseline preparation uses tiny fixed training budgets and no nested validation/tuning protocol. Duplicate aliases may also produce identical implementations as separate rows. The numerical comparison is an internal benchmark result, not a frontier-baseline comparison.

### P0-005: Evidence release is not independently reproducible

The inspected release asset verifies its own checksum and summaries but omits the referenced checkpoints and raw input file hashes. It can audit a finalization step, not reproduce training from canonical raw data.

### P0-006: Calibration is self-attested

The EEG v1 gate can mark calibration checked from a boolean while the recovered calibration artifact reports calibration unavailable. The uncertainty head is not evidence of calibrated predictive uncertainty.

### P0-007: Claim vocabulary exceeds evidence

Some documentation directions use neural field, neural operator, digital twin, foundation model, symptom map, causal, or clinical language without matching experiments. Claim gates block many of these, but generated report success can still be mistaken for scientific support.

## P1 Findings

- Split manifests do not reject duplicate record identifiers before dictionary/hash construction.
- Session grouping can use a bare session label such as `session-0`, potentially collapsing sessions across subjects.
- Time splits lack a mandatory embargo and may mix subjects.
- MOABB manifests do not preserve raw source checksums, complete preprocessing parameters, reference/filter history, or dependency-resolved paradigm defaults.
- A stale editable installation can silently import code from another Kahlus worktree when `PYTHONPATH=src` is omitted.
- Broad dependency bounds and no lockfile make local/A100 environments materially different.
- The 7-GPU run history includes rank drift and NCCL timeout; a 3-GPU finalization artifact does not validate long distributed training.
- `tests/forecastability/test_m2.py:97` performs a live PhysioNet request in the default test suite. The audit run timed out after 20 seconds, leaving an otherwise passing suite red. Network integration checks must be opt-in and use a pinned local response for the default suite.

## P2/P3 Findings

- NFC selects the first available modality rather than fusing all supplied modalities.
- The Transformer backbone contains no explicit positional encoding.
- `tiny_ssm`, `ssm_fallback`, and `ssm` are GRU aliases.
- Uncertainty uses a positive head but no proper probabilistic objective or calibration protocol.
- `tests/eeg_v1/test_eeg_v1_sprint_a.py` exceeds 6,000 lines and `eeg_v1/reporting.py` exceeds 2,700, concentrating assumptions and review risk.
- A duplicate unreachable return exists in the persistence baseline path.

## Security and Destructive Behavior

No obvious secret exfiltration, broad destructive deletion, or unsafe shell interpolation was found in the sampled high-risk paths. This does not replace a dedicated supply-chain audit.

## Fix Order

Do not tune models first. Replace the task definition, write a regression test demonstrating zero context-target overlap, invalidate historical future-window labels, then rebuild statistics, baselines, and evidence from raw manifests.
