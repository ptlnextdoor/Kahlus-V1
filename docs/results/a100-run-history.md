# A100 Run History

This is the tracked GitHub-friendly ledger for imported A100 evidence bundles. The source zip archives are kept under `outputs/past-runs-import/`, which is ignored by Git; this file and `a100-run-history.csv` preserve the durable record that should live in the repository.

The ledger intentionally records only lightweight metadata, hashes, configuration identity, gate status, and headline metrics. It does not include raw neural data, prepared manifests, checkpoints, or full logs.

For multi-task runs, `Test MSE` and `Test Pearson r` are the `metrics.json` headline values. Task-level sidecar values must be reported with their task name.

## Imported Archives

| Label | Commit | Experiment | Kind | Status | GPUs | Steps | Best Task | Best Val MSE | Headline Scope | Test MSE | Test Pearson r | Claim Allowed |
| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: | ---: | --- |
| `423248a` | `423248a119463581c4526ccfb4e35b47de46d121` | `moabb_a100_smoke` | A100 smoke | `completed_prepared_training` | 1 | 50 | `future_state_forecasting` | 39.819112 | `metrics_json_headline` | 46.233087 | 0.376921 | false |
| `7de1e50` | `7de1e5081591ad1a763a37c0211f84ac677f3e73` | `moabb_a100_smoke` | A100 smoke | `completed_prepared_training` | 1 | 50 | `future_state_forecasting` | 39.819112 | `metrics_json_headline` | 46.233087 | 0.376921 | false |
| `82e2f0b` | `82e2f0b4baba8796c09ec27e98592db53ad3a002` | `moabb_a100_smoke` | A100 smoke | `completed_prepared_training` | 6 | 50 | `future_state_forecasting` | 38.411555 | `metrics_json_headline` | 43.449179 | 0.458321 | false |
| `822a80e` | `822a80e47f87e59fbc8f9cea9b1a15cada4b6c73` | `moabb_a100` | A100 full | `failed_6gpu_torchrun_training` | 6 | 50000 configured | n/a | n/a | n/a | n/a | n/a | n/a |
| `6621642-ddpfix` | `66216425136fed888a8511d298c3cf3fbc877987` | `moabb_a100_3gpu_batchmatched_ddpfix_resume` | A100 DDP finalize recovery | `completed_prepared_training` | 3 | 100000 | `future_state_forecasting` | 2.228651 | `aggregate_multitask_metrics_json` | 28.546603 | 0.479801 | false |

## Task-Level Sidecars

| Label | Task | Test MSE | Test Pearson r | Interpretation |
| --- | --- | ---: | ---: | --- |
| `6621642-ddpfix` | `future_state_forecasting` | 3.116075 | 0.972108 | Narrow task-level forecasting result only. |
| `6621642-ddpfix` | `masked_neural_reconstruction` | 53.977132 | -0.012507 | Failed reconstruction task; blocks broad neural-translation claims. |

## Archive Hashes

| Label | Source Archive | SHA256 |
| --- | --- | --- |
| `423248a` | `neurotwin-a100-results-423248a-evidence.zip` | `28acd60971ca1deb4adc8403cc6156997594f8194ce2239bb4de730189d9ec31` |
| `7de1e50` | `neurotwin-a100-results-7de1e50-evidence.zip` | `be3fff67b784898ec80664a25b24b80d1f6f74dbe5712f619ed1a817aac800b9` |
| `82e2f0b` | `neurotwin-a100-results-82e2f0b-evidence.zip` | `dd4761cc3ae2e46d0f0527392081f0315b652a7b676d800822153d21f686aa28` |
| `822a80e` | `neurotwin-a100-results-822a80e-failure-evidence.zip` | `64c0568c16dcbba9ecb3fa98ae434694a2c46bf4efb9ca4be3d06b73d9c9b7ae` |
| `6621642-ddpfix` | `neurotwin-6621642-ddpfix-resume-handoff.zip` | `196ee135ea1b5f7595b886ed64862a51e156ff84c229b5e625452e1751006b47` |

Release artifact for `6621642-ddpfix`: <https://github.com/ptlnextdoor/Kahlus-V1/releases/tag/evidence-6621642-ddpfix-resume>

## Interpretation Notes

- These completed runs are MOABB A100 infrastructure smoke runs, not scientific evidence of model superiority.
- `scientific_claim_allowed=false` for the completed smoke runs.
- The `822a80e` full run passed the paper-mode gate before training, then failed during 6-GPU torchrun training before `summary.json` or `metrics.json` was produced.
- `6621642-ddpfix` is recovered engineering evidence: the model reached the 100k-step checkpoint, then a patched 3-GPU zero-step resume proved the final distributed path can exit cleanly and write artifacts. It is MOABB EEG evidence, not Algonauts stimulus-to-fMRI evidence.
- The 3.116075 MSE / 0.972108 Pearson result for `6621642-ddpfix` is `future_state_forecasting` only. The whole-run headline remains the aggregate multi-task `metrics.json` value, 28.546603 MSE / 0.479801 Pearson, and reconstruction failed.
- The CSV is the canonical machine-readable ledger; this Markdown file is the GitHub-readable summary.
