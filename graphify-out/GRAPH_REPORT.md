# Graph Report - hyderabad  (2026-05-30)

## Corpus Check
- 73 files · ~37,094 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 576 nodes · 1208 edges · 31 communities detected
- Extraction: 63% EXTRACTED · 37% INFERRED · 0% AMBIGUOUS · INFERRED: 442 edges (avg confidence: 0.74)
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
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]

## God Nodes (most connected - your core abstractions)
1. `run_prepared_training()` - 29 edges
2. `NeuralStateSpaceTranslator` - 28 edges
3. `NeuralEventBatch` - 28 edges
4. `_cmd_train()` - 23 edges
5. `audit_prepared_eval_inputs()` - 21 edges
6. `build_split_manifest()` - 20 edges
7. `RecordingRecord` - 18 edges
8. `_cmd_data_prepare()` - 17 edges
9. `_cmd_data_smoke()` - 17 edges
10. `make_synthetic_recordings()` - 17 edges

## Surprising Connections (you probably didn't know these)
- `AdapterTests` --uses--> `MissingOptionalDependency`  [INFERRED]
  tests/test_adapters_moabb_bids.py → src/neurotwin/adapters/moabb.py
- `ModelMetadataGeometryTests` --uses--> `NeuralStateSpaceTranslator`  [INFERRED]
  tests/test_model_metadata_geometry.py → src/neurotwin/models/torch_models.py
- `WindowsAndMoabbLoaderTests` --uses--> `WindowSpec`  [INFERRED]
  tests/test_windows_and_moabb_loader.py → src/neurotwin/data/windows.py
- `BaselinesAndArchitectureTests` --uses--> `NumpyRidgeBaseline`  [INFERRED]
  tests/test_baselines_and_architecture.py → src/neurotwin/models/baselines.py
- `BaselinesAndArchitectureTests` --uses--> `TorchMLPBaseline`  [INFERRED]
  tests/test_baselines_and_architecture.py → src/neurotwin/models/baselines.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (44): _baseline_catalog(), BaselineFailure, _cross_modal_task(), _fit_neurotwin(), _fit_ridge(), _fit_torch_sequence_model(), _flatten_time(), _future_task() (+36 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (30): audit_prepared_eval_inputs(), _prepared_windows_by_split(), PreparedEvalAuditReport, Audit prepared eval inputs before any benchmark score is trusted., Audit prepared eval inputs before any benchmark score is trusted., _record_id(), _window_overlap_violations(), event_manifest_summary() (+22 more)

### Community 2 - "Community 2"
Cohesion: 0.08
Nodes (33): format_prepared_eval_audit(), _metrics(), _cmd_eval(), bandpower_error(), bootstrap_ci(), mae(), mse(), pearsonr() (+25 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (33): _cmd_cluster_preflight(), _cmd_estimate(), _cmd_train(), _config_value(), _has_prepared_training_inputs(), _parse_split_windows(), ConfigError, load_config() (+25 more)

### Community 4 - "Community 4"
Cohesion: 0.08
Nodes (30): audit_split_manifest(), AuditReport, _forbidden_metadata(), _metadata_group_overlap(), _window_overlap(), _cmd_split_audit(), check_manifest_leakage(), LeakageReport (+22 more)

### Community 5 - "Community 5"
Cohesion: 0.09
Nodes (25): cleanup_process_group(), DistributedInfo, get_distributed_info(), get_rank_metrics_path(), maybe_init_process_group(), unwrap_model(), wrap_ddp_if_initialized(), _aggregate_task_results() (+17 more)

### Community 6 - "Community 6"
Cohesion: 0.14
Nodes (24): _cmd_data_prepare(), _cmd_data_smoke(), save_data_manifest(), save_leakage_report(), save_split_manifest(), balanced_trial_subset(), _build_moabb_paradigm(), _build_record_id() (+16 more)

### Community 7 - "Community 7"
Cohesion: 0.13
Nodes (21): _cmd_report(), competitor_registry(), CompetitorSpec, _adaptation_rows(), _csv_cell(), _csv_rows(), _flatten_metrics(), generate_compare_report() (+13 more)

### Community 8 - "Community 8"
Cohesion: 0.16
Nodes (11): _cmd_cluster_materialize_config(), ClusterMaterializeConfigReport, ClusterPreflightReport, _config_value(), format_cluster_materialize_config(), materialize_cluster_config(), Write a cluster config with absolute prepared-manifest paths., Validate cluster launch inputs before an expensive SLURM allocation runs. (+3 more)

### Community 9 - "Community 9"
Cohesion: 0.18
Nodes (20): build_prepared_window_tasks(), _cross_modal_task_from_windows(), _dataset_site_generalization_from_windows(), _first_modality_with_splits(), _future_task_from_windows(), _future_xy(), _group_windows(), _masked_task_from_windows() (+12 more)

### Community 10 - "Community 10"
Cohesion: 0.17
Nodes (18): bids_manifest_summary(), _events_for(), _infer_modality(), _is_bids_signal(), _listlike(), _load_timeseries_derivative(), _parse_entities(), _read_tsv() (+10 more)

### Community 11 - "Community 11"
Cohesion: 0.14
Nodes (7): _build_backbone(), _build_modality_encoder(), _check_sequence_tensor(), _LinearModalityEncoder, _SSMFallbackBackbone, _TemporalConvModalityEncoder, _TransformerBackbone

### Community 12 - "Community 12"
Cohesion: 0.17
Nodes (1): ResearchArtifactTests

### Community 13 - "Community 13"
Cohesion: 0.2
Nodes (7): dataset_registry(), DatasetAdapterSpec, RegistryTests, permissive_upstreams(), quarantined_upstreams(), upstream_registry(), UpstreamSpec

### Community 14 - "Community 14"
Cohesion: 0.28
Nodes (1): MoabbCliAndGeneralizationTests

### Community 15 - "Community 15"
Cohesion: 0.22
Nodes (1): Data schemas, split manifests, and leakage guards.

### Community 16 - "Community 16"
Cohesion: 0.33
Nodes (6): _cmd_doctor(), _check_runs_writable(), DoctorCheck, DoctorReport, format_doctor_report(), run_doctor()

### Community 17 - "Community 17"
Cohesion: 0.43
Nodes (1): ExpandedCliTests

### Community 18 - "Community 18"
Cohesion: 0.53
Nodes (1): ManifestAuditAndTorchrunTests

### Community 19 - "Community 19"
Cohesion: 0.83
Nodes (3): _flatten_metrics(), main(), _read_summary()

### Community 20 - "Community 20"
Cohesion: 0.5
Nodes (2): NeuralStateSpaceTranslatorConfig, Configuration shell for the future NeuroTwin model implementation.

### Community 21 - "Community 21"
Cohesion: 0.67
Nodes (1): CliReportTests

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (2): _clone_or_checkout(), main()

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): Validate cluster launch inputs before an expensive SLURM allocation runs.

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): Per-timepoint MLP baseline for neural windows.

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): Small Conv1D/TCN-style baseline with stable [batch, time, feature] IO.

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (1): Small Transformer baseline for CPU shape and smoke tests.

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): CPU debug stand-in for long-sequence SSM baselines until Mamba is pinned.

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): Modality-tokenizer + shared latent dynamics + modality-readout scaffold.

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): Encode observed modalities into shared latent neural tokens.

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): Encode observed modalities into shared latent neural tokens.

## Knowledge Gaps
- **24 isolated node(s):** `Raised when an experiment config cannot be loaded or validated.`, `Resolve a source commit from git, falling back to COMMIT_HASH.txt.`, `Validate cluster launch inputs before an expensive SLURM allocation runs.`, `Write a cluster config with absolute prepared-manifest paths.`, `Closed-form ridge baseline for sanity checks and tiny CPU benchmarks.` (+19 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 12`** (15 nodes): `ResearchArtifactTests`, `._assert_runner_archive()`, `._copy_repo_to_temp_git()`, `.test_a100_h100_configs_scripts_and_paper_docs_exist()`, `.test_a100_slurm_scripts_require_safe_inputs()`, `.test_chapman_first_run_launcher_contains_required_sequence()`, `.test_claims_doc_blocks_forbidden_claims()`, `.test_moabb_benchmark_script_blocks_slurm_tmp_fallback()`, `.test_moabb_scripts_and_cluster_configs_use_benchmark_windows()`, `.test_operator_run_bundle_files_are_self_contained()`, `.test_package_a100_handoff_zip_smokes_real_archive()`, `.test_package_bundle_uses_head_archive_and_dirty_guard()`, `.test_package_runner_bundle_smokes_real_archive()`, `.test_runpod_rehearsal_is_budget_gated()`, `test_research_artifacts.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 14`** (9 nodes): `MoabbCliAndGeneralizationTests`, `.run_cli_no_check()`, `.test_dataset_site_generalization_task()`, `.test_moabb_prepare_missing_deps_fails_cleanly()`, `.test_moabb_smoke_missing_deps_fails_cleanly()`, `.test_neural_translation_suite_includes_generalization()`, `test_real_moabb_benchmark_script_creates_windows_and_tasks()`, `test_real_moabb_smoke_script_creates_windows()`, `test_moabb_cli_and_generalization.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 15`** (9 nodes): `Data schemas, split manifests, and leakage guards.`, `__init__.py`, `__init__.py`, `__init__.py`, `__init__.py`, `__init__.py`, `__init__.py`, `__init__.py`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 17`** (8 nodes): `ExpandedCliTests`, `.run_cli()`, `.test_bids_prepare_writes_event_manifest_when_derivative_exists()`, `.test_data_and_split_audits()`, `.test_estimate_and_train_dry_run()`, `.test_report_compare_writes_aggregate_artifacts()`, `.test_report_run_dir()`, `test_cli_expanded.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 18`** (6 nodes): `ManifestAuditAndTorchrunTests`, `.run_cli()`, `.test_rank_zero_writes_shared_artifacts_and_rank_metrics()`, `.test_split_audit_from_saved_manifest()`, `.test_train_rank_one_does_not_write_shared_checkpoint()`, `test_manifest_audit_and_torchrun.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (4 nodes): `translator.py`, `NeuralStateSpaceTranslatorConfig`, `.describe()`, `Configuration shell for the future NeuroTwin model implementation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (3 nodes): `CliReportTests`, `.test_report_mentions_corrected_boss_fight_and_split_rules()`, `test_cli_report.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (3 nodes): `vendor_upstreams.py`, `_clone_or_checkout()`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `Validate cluster launch inputs before an expensive SLURM allocation runs.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `Per-timepoint MLP baseline for neural windows.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `Small Conv1D/TCN-style baseline with stable [batch, time, feature] IO.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `Small Transformer baseline for CPU shape and smoke tests.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `CPU debug stand-in for long-sequence SSM baselines until Mamba is pinned.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `Modality-tokenizer + shared latent dynamics + modality-readout scaffold.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Encode observed modalities into shared latent neural tokens.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `Encode observed modalities into shared latent neural tokens.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_cmd_train()` connect `Community 3` to `Community 0`, `Community 1`, `Community 4`, `Community 5`, `Community 6`?**
  _High betweenness centrality (0.145) - this node is a cross-community bridge._
- **Why does `run_prepared_training()` connect `Community 5` to `Community 0`, `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 9`?**
  _High betweenness centrality (0.144) - this node is a cross-community bridge._
- **Why does `NeuralStateSpaceTranslator` connect `Community 0` to `Community 11`, `Community 5`?**
  _High betweenness centrality (0.113) - this node is a cross-community bridge._
- **Are the 19 inferred relationships involving `run_prepared_training()` (e.g. with `.test_prepared_training_writes_checkpoint_and_metrics()` and `.test_prepared_training_runs_all_neural_translation_tasks()`) actually correct?**
  _`run_prepared_training()` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Are the 21 inferred relationships involving `NeuralStateSpaceTranslator` (e.g. with `ModelMetadataGeometryTests` and `BaselinesAndArchitectureTests`) actually correct?**
  _`NeuralStateSpaceTranslator` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `NeuralEventBatch` (e.g. with `NeuralEventBatchTests` and `Create paired synthetic recordings that mirror the recording manifest.`) actually correct?**
  _`NeuralEventBatch` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `_cmd_train()` (e.g. with `load_config()` and `estimate_config()`) actually correct?**
  _`_cmd_train()` has 20 INFERRED edges - model-reasoned connections that need verification._