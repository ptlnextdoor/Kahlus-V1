# Track B Algonauts Benchmark Protocol

This document defines the reviewer-facing protocol for the Kahlus/NeuroTwin Track B Algonauts 2025 benchmark. The goal is not to claim state of the art by default. The goal is to make the Pair-Operator/NFC architecture falsifiable against strong leakage-safe baselines before spending a full six-A100 run.

## Dataset And Claim Scope

Primary dataset: Algonauts 2025 / CNeuroMod movie stimulus to fMRI.

Required data contract:

- Four subjects: `sub-01`, `sub-02`, `sub-03`, `sub-05`.
- fMRI modality only for Track B first pass.
- Each prepared response array is shaped `[time,1000]` over Schaefer 1000 parcels.
- Stimulus features are aligned one row per fMRI TR, with `stimulus_embedding.shape[0] == signal.shape[0]`.
- Raw or precomputed feature artifacts have a local source path and SHA256 hash.
- `stimulus_to_fmri_response` must be present as a real prepared task, not skipped.

Claim eligibility:

- Real stimulus claims require `require_real_stimulus=true`.
- Feature sources must be real precomputed stimulus artifacts. Hash-only transcript features, synthetic embeddings, placeholders, and self-attested embeddings are plumbing-only.
- The prepared-task evidence gate must verify that `stimulus_feature_hash` matches the referenced source artifact.
- No raw neural data, movie files, checkpoints, credentials, or downloaded datasets are committed to git.

## Splits

All splits are at whole movie, episode, or run level. Random TR/window splits are prohibited.

The local `official` adapter policy is official-compatible:

- If official/test/OOD-labeled response partitions are present, they are used.
- If withheld official responses are unavailable locally, the adapter uses a whole-run local-dev partition:
  - Friends seasons 1-5: train.
  - Friends season 6: validation.
  - Movie10, explicit test, or OOD-labeled runs: test.

The split manifest must preserve the exact record IDs, source hashes, split assignments, and stimulus IDs. Any boundary-buffer or HRF-delay exclusions used by a later feature pipeline must be recorded in the feature/preprocessing manifests.

## Debug Gate

The one-GPU debug gate must pass before any six-GPU sweep.

Required checks:

- Algonauts raw/prepared data are present under an approved cluster root.
- The four expected subjects are present.
- fMRI arrays are finite and 1000-parcel compatible.
- Stimulus features are nonzero, finite, source-path-backed, and SHA256 verified.
- `event_manifest.json`, `split_manifest.json`, `data_manifest.json`, `feature_manifest.json`, `stimulus_manifest.json`, `leakage_report.json`, and `eval_audit.json` exist.
- `eval_audit.json.passed == true`.
- The prepared baseline suite emits finite metrics for `stimulus_to_fmri_response`.
- `paper_mode_gate.json` passes for seeds `0,1,2` on the debug baseline run.
- A short Kahlus debug model run exits cleanly and writes `summary.json`, `metrics.csv`, and final evidence artifacts.

Failure rule: if any required debug check fails, stop. Do not launch the six-GPU sweep.

## Baselines

Minimum local baselines on the exact same prepared tensors:

- `linear_ridge`: primary ridge anchor.
- `autoregressive_ridge`: history-aware ridge anchor when the task supports it.
- `persistence`, `train_mean`, and `random_permutation`: null/floor baselines.
- `neurotwin` or current direct NeuroTwin model.
- `pair_operator_no_pair_state`: direct ablation without pair state.
- `pair_operator_full`: full Pair-Operator/NFC candidate.

External systems such as TRIBE/VIBE/BrainVista are contextual references unless their exact code and feature contract are run on the same split. BrainVista-style local code is a labeled reimplementation/approximation, not exact BrainVista.

## Six-GPU Ablation Sweep

Run three seed waves: `0`, `1`, `2`.

Each wave uses six A100s, one arm per GPU:

- `ridge_anchor`
- `current_neurotwin`
- `pair_operator_no_pair_state`
- `pair_operator_low_rank_pair_state`
- `pair_operator_pair_state_uncertainty`
- `pair_operator_full`

All arms must share:

- identical split manifest hash;
- identical feature manifest hash;
- identical preprocessing policy;
- train-only scaling/PCA/normalization;
- identical metric and statistical-test code.

Changing feature extractors, split files, normalization scope, or held-out records between arms invalidates the ablation.

## Metrics

Primary metric for Algonauts Track B:

- Mean Pearson `r` for `stimulus_to_fmri_response`.

Also report:

- MSE, MAE, R2, Spearman;
- per-seed metrics;
- per-subject metrics when available;
- per-parcel/per-network summaries when available;
- ridge gap: `pair_operator_full - ridge_anchor`;
- architecture deltas:
  - `pair_operator_full - pair_operator_no_pair_state`;
  - `pair_operator_full - current_neurotwin`;
- uncertainty-error correlation for uncertainty arms.

Do not substitute forecasting or pattern-correlation metrics for the Algonauts stimulus-to-fMRI score. Those can be secondary dynamics diagnostics only.

## Strict Pass Gate

The long six-GPU Pair-Operator run is allowed only if the sweep passes all strict checks:

- `pair_operator_full` beats `pair_operator_no_pair_state` by at least `0.01` mean Pearson.
- `pair_operator_full` beats `current_neurotwin` by at least `0.01` mean Pearson.
- Positive direction holds in at least `2/3` seeds.
- The full model is not badly below ridge: `full >= ridge_anchor - 0.01`, or it beats ridge.
- Uncertainty-error correlation is positive and finite.
- No required task is quarantined.
- Evidence/model-card artifacts are clean.

If per-subject artifacts exist, the positive direction should hold in at least `3/4` subjects. If those artifacts do not exist, the reviewer gate should be considered incomplete and the paper should not make a subject-consistency claim.

## Long Run

Only after the strict gate passes:

- model: `pair_operator_full`;
- GPUs: exactly six idle A100s;
- initial steps: `50000`;
- config: `configs/train/algonauts_pair_operator_full.yaml`;
- run root: `/raid/scratch/$USER/kahlus-algonauts-trackb-v1/long` or the stage root selected in the launch script.

Codex/agents must detach after safe launch. Safe launch means:

- tmux or scheduler job is alive;
- correct GPU count and IDs are visible in the container/job;
- persistent root, prepared root, and config paths are visible;
- config materializes with absolute manifest paths;
- audit/preflight passes;
- the training command starts;
- first metrics/log row appears when applicable;
- no immediate traceback, NaN quarantine, DDP mismatch, NCCL timeout, OOM, or `ChildFailedError`.

After safe launch, create a thread heartbeat near the estimated finish time. Do not keep an agent attached while training.

## Artifact Checklist

Every claim-eligible stage must preserve:

- `data_manifest.json`;
- `split_manifest.json`;
- `event_manifest.json`;
- `feature_manifest.json`;
- `stimulus_manifest.json`;
- `leakage_report.json`;
- `eval_audit.json`;
- `baseline_ranking.csv` or `prepared_baseline_suite.json`;
- `paper_mode_gate.json` when paper mode is run;
- `strict_gate.json` for sweep decisions;
- `summary.json`, `metrics.json`, `metrics.csv`, and `metrics*.jsonl` for training runs;
- `diagnostic_report.md`, `RUN_REPORT.md`, `EEG_MODEL_CARD.md` or successor model card when available;
- container image, git commit, config YAML, and command logs.

Evidence zips must exclude raw data, checkpoints, credentials, `.env*`, tokens, SSH keys, nested archives, and tarballs.

## Interpretation Rules

- If the full model beats internal ablations but loses to ridge, the result supports internal architecture progress only. It does not support a better encoding-model claim.
- If ridge beats all Kahlus variants by a large margin, stop and write a negative/diagnostic result.
- If the debug gate fails feature verification, the run is not paper evidence.
- If the long run launches without the strict gate, it is an engineering run, not a reviewer-safe scientific run.
