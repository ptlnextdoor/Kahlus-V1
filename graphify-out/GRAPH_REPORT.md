# Graph Report - hyderabad  (2026-05-31)

## Corpus Check
- 126 files · ~75,913 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 782 nodes · 2031 edges · 41 communities detected
- Extraction: 69% EXTRACTED · 31% INFERRED · 0% AMBIGUOUS · INFERRED: 638 edges (avg confidence: 0.74)
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
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]

## God Nodes (most connected - your core abstractions)
1. `NeuralEventBatch` - 36 edges
2. `run_prepared_training()` - 31 edges
3. `NeuralStateSpaceTranslator` - 30 edges
4. `build_split_manifest()` - 29 edges
5. `audit_prepared_eval_inputs()` - 28 edges
6. `_cmd_train()` - 25 edges
7. `RecordingRecord` - 23 edges
8. `make_synthetic_recordings()` - 22 edges
9. `SupervisedWindowTask` - 22 edges
10. `_run_task_models()` - 21 edges

## Surprising Connections (you probably didn't know these)
- `AdapterTests` --uses--> `MissingOptionalDependency`  [INFERRED]
  tests/test_adapters_moabb_bids.py → src/neurotwin/adapters/moabb.py
- `BaselinesAndArchitectureTests` --uses--> `NeuralStateSpaceTranslator`  [INFERRED]
  tests/test_baselines_and_architecture.py → src/neurotwin/models/torch_models.py
- `BaselineSuiteTests` --uses--> `SupervisedWindowTask`  [INFERRED]
  tests/test_baseline_suite.py → src/neurotwin/benchmarks/baseline_suite.py
- `BaselineSuiteTests` --uses--> `TribeStyleModel`  [INFERRED]
  tests/test_baseline_suite.py → src/neurotwin/models/tribe_style.py
- `PreparedEventSuiteTests` --uses--> `PreparedSuiteConfig`  [INFERRED]
  tests/test_prepared_event_suite.py → src/neurotwin/benchmarks/prepared_suite.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.07
Nodes (40): _cmd_data_prepare(), _cmd_data_smoke(), event_manifest_summary(), _jsonable(), load_event_batches(), _loads_json(), Persist prepared event batches for offline training/eval jobs., _safe_name() (+32 more)

### Community 1 - "Community 1"
Cohesion: 0.09
Nodes (35): _baseline_catalog(), BaselineFailure, _cross_modal_task(), _fit_autoregressive_ridge(), _fit_neurotwin(), _fit_ridge(), _fit_torch_sequence_model(), _fit_tribe_style() (+27 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (30): barrier_if_distributed(), cleanup_process_group(), DistributedInfo, get_distributed_info(), get_rank_metrics_path(), is_distributed(), is_rank_zero(), maybe_init_process_group() (+22 more)

### Community 3 - "Community 3"
Cohesion: 0.1
Nodes (28): _metrics(), bandpower_error(), bootstrap_ci(), mae(), mse(), pearsonr(), r2_score(), rank_models() (+20 more)

### Community 4 - "Community 4"
Cohesion: 0.07
Nodes (20): Run tiny local baselines on paired synthetic windows.      This is intentionally, Run a tiny deterministic CPU training loop for CLI and CI smoke tests., run_synthetic_training(), TrainingSmokeResult, ModelMetadataGeometryTests, ModelShapeTests, TrainingSmokeTests, _build_backbone() (+12 more)

### Community 5 - "Community 5"
Cohesion: 0.11
Nodes (44): SupervisedWindowTask, _add_degenerate_ci(), _add_degenerate_task_metric_ci(), _aggregate_seed_metrics(), _aggregate_seed_ranks(), _aggregate_seed_tasks(), build_prepared_window_tasks(), _collect_aggregate_seed_ranks() (+36 more)

### Community 6 - "Community 6"
Cohesion: 0.1
Nodes (29): _cmd_train(), ConfigError, load_config(), Raised when an experiment config cannot be loaded or validated., require_config_keys(), append_jsonl(), capture_environment(), capture_run_metadata() (+21 more)

### Community 7 - "Community 7"
Cohesion: 0.1
Nodes (21): _cmd_cluster_materialize_config(), _cmd_cluster_preflight(), _cmd_data_audit(), _cmd_estimate(), _cmd_split_audit(), _config_value(), _has_prepared_training_inputs(), main() (+13 more)

### Community 8 - "Community 8"
Cohesion: 0.08
Nodes (14): Data schemas, split manifests, and leakage guards., _build_events(), _event_embedding(), from_checkpoint(), from_pretrained(), _load_local_config(), Build minimal local event rows for smoke/pipeline tests.          Text events ar, Compatibility shim returning local event rows, not a pandas DataFrame. (+6 more)

### Community 9 - "Community 9"
Cohesion: 0.11
Nodes (29): Audit prepared eval inputs before any benchmark score is trusted., Persist prepared event batches for offline training/eval jobs., Persist prepared event batches for offline training/eval jobs., balanced_trial_subset(), _build_moabb_paradigm(), _build_record_id(), _listlike(), load_moabb_trials() (+21 more)

### Community 10 - "Community 10"
Cohesion: 0.11
Nodes (23): format_prepared_eval_audit(), _cmd_eval(), EvalCommandConfig, EvalCommandResult, _manifest_paths(), _prepared_suite_config(), _run_audit_command(), run_eval_command() (+15 more)

### Community 11 - "Community 11"
Cohesion: 0.23
Nodes (25): _add_concrete_seed_record(), _aggregate_rank(), _audit_payload(), _ci_violations(), _ci_violations_for_payload(), _coerce_seed(), _finite_number(), _has_finite_ci() (+17 more)

### Community 12 - "Community 12"
Cohesion: 0.21
Nodes (18): bids_manifest_summary(), _events_for(), _infer_modality(), _is_bids_signal(), _listlike(), _load_timeseries_derivative(), _parse_entities(), _read_tsv() (+10 more)

### Community 13 - "Community 13"
Cohesion: 0.24
Nodes (17): audit_prepared_eval_inputs(), audit_split_manifest(), AuditReport, _duplicate_metadata_value_violations(), _event_metadata_violations(), _forbidden_event_metadata_violations(), _forbidden_metadata(), _hidden_subject_metadata_violations() (+9 more)

### Community 14 - "Community 14"
Cohesion: 0.32
Nodes (16): _cmd_report(), _adaptation_rows(), _csv_cell(), _csv_rows(), _flatten_metrics(), generate_compare_report(), generate_run_report(), generate_suite_report() (+8 more)

### Community 15 - "Community 15"
Cohesion: 0.24
Nodes (16): behavior_metadata(), dataset_id(), geometry_metadata(), n_space(), n_time(), _optional_float(), _optional_str(), preprocessing_hash() (+8 more)

### Community 16 - "Community 16"
Cohesion: 0.2
Nodes (7): dataset_registry(), DatasetAdapterSpec, RegistryTests, permissive_upstreams(), quarantined_upstreams(), upstream_registry(), UpstreamSpec

### Community 17 - "Community 17"
Cohesion: 0.27
Nodes (7): load_split_manifest(), record_from_dict(), record_to_dict(), save_data_manifest(), split_manifest_from_dict(), split_manifest_to_dict(), ManifestPersistenceAndEvalSuiteTests

### Community 18 - "Community 18"
Cohesion: 0.14
Nodes (1): ArtifactDocsContractsTests

### Community 19 - "Community 19"
Cohesion: 0.24
Nodes (4): assert_runner_archive(), copy_repo_to_temp_git(), HandoffZipArtifactTests, RunnerBundleArtifactTests

### Community 20 - "Community 20"
Cohesion: 0.42
Nodes (8): _cmd_doctor(), _check_runs_writable(), DoctorCheck, DoctorReport, format_doctor_report(), _is_writable(), passed(), run_doctor()

### Community 21 - "Community 21"
Cohesion: 0.54
Nodes (1): EvidenceBundleArtifactTests

### Community 22 - "Community 22"
Cohesion: 0.43
Nodes (1): ExpandedCliTests

### Community 23 - "Community 23"
Cohesion: 0.7
Nodes (4): main(), _nccl_version(), _payload(), _positive_int()

### Community 24 - "Community 24"
Cohesion: 0.4
Nodes (2): NeuralStateSpaceTranslatorConfig, Configuration shell for the future NeuroTwin model implementation.

### Community 25 - "Community 25"
Cohesion: 0.83
Nodes (3): _flatten_metrics(), main(), _read_summary()

### Community 26 - "Community 26"
Cohesion: 0.67
Nodes (1): CliReportTests

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (2): _clone_or_checkout(), main()

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): Load a local NeuroTwin TRIBE-style config or seeded defaults.          No pretra

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): Compatibility shim for TRIBE-style callers.          Prefer :meth:`from_checkpoi

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): Validate the artifact contract required before paper-mode claims.      This is i

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (1): Validate the artifact contract required before paper-mode claims.      This is i

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): Validate the artifact contract required before paper-mode claims.      This is i

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): Validate cluster launch inputs before an expensive SLURM allocation runs.

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (1): Per-timepoint MLP baseline for neural windows.

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): Small Conv1D/TCN-style baseline with stable [batch, time, feature] IO.

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): Small Transformer baseline for CPU shape and smoke tests.

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): CPU debug stand-in for long-sequence SSM baselines until Mamba is pinned.

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Modality-tokenizer + shared latent dynamics + modality-readout scaffold.

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): Encode observed modalities into shared latent neural tokens.

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): Encode observed modalities into shared latent neural tokens.

## Knowledge Gaps
- **36 isolated node(s):** `Raised when an experiment config cannot be loaded or validated.`, `Resolve a source commit from git, falling back to COMMIT_HASH.txt.`, `Validate cluster launch inputs before an expensive SLURM allocation runs.`, `Write a cluster config with absolute prepared-manifest paths.`, `Closed-form ridge baseline for sanity checks and tiny CPU benchmarks.` (+31 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 18`** (14 nodes): `ArtifactDocsContractsTests`, `.test_a100_h100_configs_scripts_and_paper_docs_exist()`, `.test_a100_runbook_separates_fast_and_heavy_lanes()`, `.test_a100_slurm_scripts_require_safe_inputs()`, `.test_agent_deploy_docs_and_dockerfile_are_6gpu_first()`, `.test_chapman_first_run_launcher_contains_required_sequence()`, `.test_claims_doc_blocks_forbidden_claims()`, `.test_docker_6gpu_runner_contains_required_sequence()`, `.test_moabb_benchmark_script_blocks_slurm_tmp_fallback()`, `.test_moabb_scripts_and_cluster_configs_use_benchmark_windows()`, `.test_operator_run_bundle_files_are_self_contained()`, `.test_runpod_rehearsal_is_budget_gated()`, `.test_tribe_style_does_not_become_required_dependency()`, `test_artifact_docs_contracts.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (8 nodes): `EvidenceBundleArtifactTests`, `._create_a100_evidence_fixture()`, `._package_a100_evidence_fixture()`, `.test_package_a100_evidence_bundle_excludes_checkpoints_and_secrets()`, `.test_package_a100_evidence_bundle_falls_back_to_summary_job_id()`, `.test_package_a100_evidence_bundle_unsafe_job_id_includes_no_logs()`, `.test_package_a100_evidence_bundle_without_job_id_includes_no_logs()`, `test_artifact_evidence_bundle.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (8 nodes): `ExpandedCliTests`, `.run_cli()`, `.test_bids_prepare_writes_event_manifest_when_derivative_exists()`, `.test_data_and_split_audits()`, `.test_estimate_and_train_dry_run()`, `.test_report_compare_writes_aggregate_artifacts()`, `.test_report_run_dir()`, `test_cli_expanded.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (5 nodes): `translator.py`, `translator.py`, `NeuralStateSpaceTranslatorConfig`, `.describe()`, `Configuration shell for the future NeuroTwin model implementation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (3 nodes): `CliReportTests`, `.test_report_mentions_corrected_boss_fight_and_split_rules()`, `test_cli_report.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (3 nodes): `vendor_upstreams.py`, `_clone_or_checkout()`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `Load a local NeuroTwin TRIBE-style config or seeded defaults.          No pretra`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Compatibility shim for TRIBE-style callers.          Prefer :meth:`from_checkpoi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `Validate the artifact contract required before paper-mode claims.      This is i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `Validate the artifact contract required before paper-mode claims.      This is i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `Validate the artifact contract required before paper-mode claims.      This is i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `Validate cluster launch inputs before an expensive SLURM allocation runs.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `Per-timepoint MLP baseline for neural windows.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `Small Conv1D/TCN-style baseline with stable [batch, time, feature] IO.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Small Transformer baseline for CPU shape and smoke tests.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `CPU debug stand-in for long-sequence SSM baselines until Mamba is pinned.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Modality-tokenizer + shared latent dynamics + modality-readout scaffold.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Encode observed modalities into shared latent neural tokens.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Encode observed modalities into shared latent neural tokens.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_prepared_training()` connect `Community 2` to `Community 0`, `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 17`?**
  _High betweenness centrality (0.105) - this node is a cross-community bridge._
- **Why does `_cmd_train()` connect `Community 6` to `Community 0`, `Community 2`, `Community 4`, `Community 7`, `Community 10`, `Community 13`, `Community 17`?**
  _High betweenness centrality (0.099) - this node is a cross-community bridge._
- **Why does `NeuralEventBatch` connect `Community 9` to `Community 0`, `Community 5`, `Community 12`, `Community 13`, `Community 15`?**
  _High betweenness centrality (0.089) - this node is a cross-community bridge._
- **Are the 41 inferred relationships involving `ValueError` (e.g. with `generate_suite_report()` and `run_prepared_training()`) actually correct?**
  _`ValueError` has 41 INFERRED edges - model-reasoned connections that need verification._
- **Are the 32 inferred relationships involving `NeuralEventBatch` (e.g. with `NeuralEventBatchTests` and `Create paired synthetic recordings that mirror the recording manifest.`) actually correct?**
  _`NeuralEventBatch` has 32 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `run_prepared_training()` (e.g. with `.test_prepared_training_writes_checkpoint_and_metrics()` and `.test_prepared_training_runs_all_neural_translation_tasks()`) actually correct?**
  _`run_prepared_training()` has 20 INFERRED edges - model-reasoned connections that need verification._
- **Are the 22 inferred relationships involving `NeuralStateSpaceTranslator` (e.g. with `ModelMetadataGeometryTests` and `BaselinesAndArchitectureTests`) actually correct?**
  _`NeuralStateSpaceTranslator` has 22 INFERRED edges - model-reasoned connections that need verification._