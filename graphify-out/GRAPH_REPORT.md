# Graph Report - /Users/aayu/conductor/workspaces/kahlus-v1/hyderabad  (2026-06-01)

## Corpus Check
- 128 files · ~151,540 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 852 nodes · 2207 edges · 41 communities detected
- Extraction: 69% EXTRACTED · 31% INFERRED · 0% AMBIGUOUS · INFERRED: 684 edges (avg confidence: 0.75)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]

## God Nodes (most connected - your core abstractions)
1. `NeuralEventBatch` - 36 edges
2. `run_prepared_training()` - 31 edges
3. `NeuralStateSpaceTranslator` - 31 edges
4. `build_split_manifest()` - 30 edges
5. `audit_prepared_eval_inputs()` - 28 edges
6. `_cmd_train()` - 25 edges
7. `make_synthetic_recordings()` - 23 edges
8. `RecordingRecord` - 23 edges
9. `SupervisedWindowTask` - 22 edges
10. `save_event_batches()` - 22 edges

## Surprising Connections (you probably didn't know these)
- `Persist prepared event batches for offline training/eval jobs.` --uses--> `NeuralEventBatch`  [INFERRED]
  src/neurotwin/data/event_io.py → /Users/aayu/conductor/workspaces/kahlus-v1/hyderabad/src/neurotwin/data/schemas.py
- `NeuralEventBatch` --uses--> `Audit prepared eval inputs before any benchmark score is trusted.`  [INFERRED]
  /Users/aayu/conductor/workspaces/kahlus-v1/hyderabad/src/neurotwin/data/schemas.py → src/neurotwin/eval/audit.py
- `Data schemas, split manifests, and leakage guards.` --uses--> `TribeStyleModel`  [INFERRED]
  src/neurotwin/data/__init__.py → /Users/aayu/conductor/workspaces/kahlus-v1/hyderabad/src/neurotwin/models/tribe_style.py
- `Data schemas, split manifests, and leakage guards.` --uses--> `TribeStyleStimulusEncoder`  [INFERRED]
  src/neurotwin/data/__init__.py → /Users/aayu/conductor/workspaces/kahlus-v1/hyderabad/src/neurotwin/models/tribe_style.py
- `Run a tiny deterministic CPU training loop for CLI and CI smoke tests.` --uses--> `NeuralStateSpaceTranslator`  [INFERRED]
  src/neurotwin/training/smoke.py → /Users/aayu/conductor/workspaces/kahlus-v1/hyderabad/src/neurotwin/models/torch_models.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (50): _cmd_cluster_preflight(), _cmd_data_audit(), _cmd_data_prepare(), _cmd_data_smoke(), _cmd_estimate(), _cmd_eval(), _cmd_split_audit(), _cmd_train() (+42 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (54): _baseline_catalog(), BaselineFailure, _cross_modal_task(), _fit_autoregressive_ridge(), _fit_neurotwin(), _fit_ridge(), _fit_torch_sequence_model(), _fit_tribe_style() (+46 more)

### Community 2 - "Community 2"
Cohesion: 0.07
Nodes (47): bids_manifest_summary(), _events_for(), _infer_modality(), _is_bids_signal(), _listlike(), _load_timeseries_derivative(), _parse_entities(), _read_tsv() (+39 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (33): _metrics(), bandpower_error(), bootstrap_ci(), mae(), mse(), pearsonr(), r2_score(), rank_models() (+25 more)

### Community 4 - "Community 4"
Cohesion: 0.06
Nodes (26): Run a tiny deterministic CPU training loop for CLI and CI smoke tests., run_synthetic_training(), TrainingSmokeResult, _latent(), make_synthetic_event_batch(), make_synthetic_multimodal_event_batches(), make_synthetic_multimodal_recordings(), _multimodal_sampling_rates() (+18 more)

### Community 5 - "Community 5"
Cohesion: 0.09
Nodes (30): barrier_if_distributed(), cleanup_process_group(), DistributedInfo, get_distributed_info(), get_rank_metrics_path(), is_distributed(), is_rank_zero(), maybe_init_process_group() (+22 more)

### Community 6 - "Community 6"
Cohesion: 0.09
Nodes (26): _bundle_rel_parts(), copy_bundle_file(), copy_current_docker_log(), copy_current_run_logs(), copy_source_file(), copy_tree_files(), current_docker_log_path(), current_slurm_job_id() (+18 more)

### Community 7 - "Community 7"
Cohesion: 0.1
Nodes (27): ConfigError, load_config(), Raised when an experiment config cannot be loaded or validated., require_config_keys(), append_jsonl(), capture_environment(), capture_run_metadata(), checkpoint_manifest() (+19 more)

### Community 8 - "Community 8"
Cohesion: 0.13
Nodes (36): main(), _flatten_metrics(), main(), _add_concrete_seed_record(), _aggregate_rank(), _audit_payload(), _ci_violations(), _ci_violations_for_payload() (+28 more)

### Community 9 - "Community 9"
Cohesion: 0.16
Nodes (36): _add_degenerate_ci(), _add_degenerate_task_metric_ci(), _aggregate_seed_metrics(), _aggregate_seed_ranks(), _aggregate_seed_tasks(), build_prepared_window_tasks(), _collect_aggregate_seed_ranks(), _collect_concrete_seed_ranks() (+28 more)

### Community 10 - "Community 10"
Cohesion: 0.14
Nodes (27): audit_prepared_eval_inputs(), _duplicate_metadata_value_violations(), _event_metadata_violations(), _forbidden_event_metadata_violations(), format_prepared_eval_audit(), _hidden_subject_metadata_violations(), _prepared_windows_by_split(), PreparedEvalAuditReport (+19 more)

### Community 11 - "Community 11"
Cohesion: 0.15
Nodes (21): _cmd_report(), competitor_registry(), CompetitorSpec, _adaptation_rows(), _csv_cell(), _csv_rows(), _flatten_metrics(), generate_compare_report() (+13 more)

### Community 12 - "Community 12"
Cohesion: 0.16
Nodes (12): _cmd_cluster_materialize_config(), ClusterMaterializeConfigReport, ClusterPreflightReport, _config_value(), format_cluster_materialize_config(), format_cluster_preflight(), materialize_cluster_config(), Write a cluster config with absolute prepared-manifest paths. (+4 more)

### Community 13 - "Community 13"
Cohesion: 0.13
Nodes (1): ArtifactDocsContractsTests

### Community 14 - "Community 14"
Cohesion: 0.24
Nodes (16): behavior_metadata(), dataset_id(), geometry_metadata(), n_space(), n_time(), _optional_float(), _optional_str(), preprocessing_hash() (+8 more)

### Community 15 - "Community 15"
Cohesion: 0.12
Nodes (1): Data schemas, split manifests, and leakage guards.

### Community 16 - "Community 16"
Cohesion: 0.23
Nodes (2): ExpandedCliTests, _valid_paper_mode_gate()

### Community 17 - "Community 17"
Cohesion: 0.2
Nodes (7): dataset_registry(), DatasetAdapterSpec, RegistryTests, permissive_upstreams(), quarantined_upstreams(), upstream_registry(), UpstreamSpec

### Community 18 - "Community 18"
Cohesion: 0.24
Nodes (4): assert_runner_archive(), copy_repo_to_temp_git(), HandoffZipArtifactTests, RunnerBundleArtifactTests

### Community 19 - "Community 19"
Cohesion: 0.42
Nodes (8): _cmd_doctor(), _check_runs_writable(), DoctorCheck, DoctorReport, format_doctor_report(), _is_writable(), passed(), run_doctor()

### Community 20 - "Community 20"
Cohesion: 0.43
Nodes (1): ManifestAuditAndTorchrunTests

### Community 21 - "Community 21"
Cohesion: 0.67
Nodes (5): audit_split_manifest(), AuditReport, _forbidden_metadata(), _metadata_group_overlap(), _window_overlap()

### Community 22 - "Community 22"
Cohesion: 0.53
Nodes (1): ManifestPersistenceAndEvalSuiteTests

### Community 23 - "Community 23"
Cohesion: 0.7
Nodes (4): main(), _nccl_version(), _payload(), _positive_int()

### Community 24 - "Community 24"
Cohesion: 0.4
Nodes (2): NeuralStateSpaceTranslatorConfig, Configuration shell for the future NeuroTwin model implementation.

### Community 25 - "Community 25"
Cohesion: 0.67
Nodes (1): CliReportTests

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (2): _clone_or_checkout(), main()

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (0): 

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (1): Load a local NeuroTwin TRIBE-style config or seeded defaults.          No pretra

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): Compatibility shim for TRIBE-style callers.          Prefer :meth:`from_checkpoi

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): Validate the artifact contract required before paper-mode claims.      This is i

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): Validate the artifact contract required before paper-mode claims.      This is i

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): Validate the artifact contract required before paper-mode claims.      This is i

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (1): Validate cluster launch inputs before an expensive SLURM allocation runs.

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): Per-timepoint MLP baseline for neural windows.

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): Small Conv1D/TCN-style baseline with stable [batch, time, feature] IO.

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (1): Small Transformer baseline for CPU shape and smoke tests.

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): CPU debug stand-in for long-sequence SSM baselines until Mamba is pinned.

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): Modality-tokenizer + shared latent dynamics + modality-readout scaffold.

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Encode observed modalities into shared latent neural tokens.

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Encode observed modalities into shared latent neural tokens.

## Knowledge Gaps
- **37 isolated node(s):** `Raised when an experiment config cannot be loaded or validated.`, `Resolve a source commit from git, falling back to COMMIT_HASH.txt.`, `Validate cluster launch inputs before an expensive SLURM allocation runs.`, `Write a cluster config with absolute prepared-manifest paths.`, `Closed-form ridge baseline for sanity checks and tiny CPU benchmarks.` (+32 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 27`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `Load a local NeuroTwin TRIBE-style config or seeded defaults.          No pretra`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `Compatibility shim for TRIBE-style callers.          Prefer :meth:`from_checkpoi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `Validate the artifact contract required before paper-mode claims.      This is i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Validate the artifact contract required before paper-mode claims.      This is i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `Validate the artifact contract required before paper-mode claims.      This is i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `Validate cluster launch inputs before an expensive SLURM allocation runs.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `Per-timepoint MLP baseline for neural windows.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `Small Conv1D/TCN-style baseline with stable [batch, time, feature] IO.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `Small Transformer baseline for CPU shape and smoke tests.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `CPU debug stand-in for long-sequence SSM baselines until Mamba is pinned.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Modality-tokenizer + shared latent dynamics + modality-readout scaffold.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Encode observed modalities into shared latent neural tokens.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Encode observed modalities into shared latent neural tokens.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_prepared_training()` connect `Community 5` to `Community 0`, `Community 9`, `Community 3`, `Community 4`?**
  _High betweenness centrality (0.101) - this node is a cross-community bridge._
- **Why does `_require_bundle_rel()` connect `Community 6` to `Community 3`?**
  _High betweenness centrality (0.098) - this node is a cross-community bridge._
- **Why does `_cmd_train()` connect `Community 0` to `Community 10`, `Community 4`, `Community 5`, `Community 7`?**
  _High betweenness centrality (0.091) - this node is a cross-community bridge._
- **Are the 43 inferred relationships involving `ValueError` (e.g. with `_require_bundle_rel()` and `generate_suite_report()`) actually correct?**
  _`ValueError` has 43 INFERRED edges - model-reasoned connections that need verification._
- **Are the 32 inferred relationships involving `NeuralEventBatch` (e.g. with `NeuralEventBatchTests` and `Create paired synthetic recordings that mirror the recording manifest.`) actually correct?**
  _`NeuralEventBatch` has 32 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `run_prepared_training()` (e.g. with `.test_prepared_training_writes_checkpoint_and_metrics()` and `.test_prepared_training_runs_all_neural_translation_tasks()`) actually correct?**
  _`run_prepared_training()` has 20 INFERRED edges - model-reasoned connections that need verification._
- **Are the 23 inferred relationships involving `NeuralStateSpaceTranslator` (e.g. with `ModelMetadataGeometryTests` and `BaselinesAndArchitectureTests`) actually correct?**
  _`NeuralStateSpaceTranslator` has 23 INFERRED edges - model-reasoned connections that need verification._