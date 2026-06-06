# Graph Report - /Users/aayu/.codex/worktrees/66ca/Kahlus-V1  (2026-06-05)

## Corpus Check
- 143 files · ~314,449 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1415 nodes · 3484 edges · 40 communities detected
- Extraction: 61% EXTRACTED · 39% INFERRED · 0% AMBIGUOUS · INFERRED: 1361 edges (avg confidence: 0.7)
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

## God Nodes (most connected - your core abstractions)
1. `NeuroTwinPairOperator` - 48 edges
2. `NeuralStateSpaceTranslator` - 46 edges
3. `NeuralStateSpaceTranslatorConfig` - 40 edges
4. `Data schemas, split manifests, and leakage guards.` - 39 edges
5. `NeuroTwinPairOperatorConfig` - 37 edges
6. `build_split_manifest()` - 36 edges
7. `SupervisedWindowTask` - 35 edges
8. `NumpyRidgeBaseline` - 32 edges
9. `PreparedTrainingTests` - 31 edges
10. `PreparedSuiteConfig` - 31 edges

## Surprising Connections (you probably didn't know these)
- `run_prepared_training()` --calls--> `cleanup_process_group()`  [INFERRED]
  /Users/aayu/.codex/worktrees/66ca/Kahlus-V1/src/neurotwin/training/prepared.py → /Users/aayu/.codex/worktrees/66ca/Kahlus-V1/src/neurotwin/runtime/distributed.py
- `_run_neural_translation_v1_command()` --calls--> `format_neural_translation_v1_report()`  [INFERRED]
  /Users/aayu/.codex/worktrees/66ca/Kahlus-V1/src/neurotwin/eval/command.py → /Users/aayu/.codex/worktrees/66ca/Kahlus-V1/src/neurotwin/benchmarks/suite.py
- `run_prepared_eval_command()` --calls--> `format_paper_mode_gate()`  [INFERRED]
  /Users/aayu/.codex/worktrees/66ca/Kahlus-V1/src/neurotwin/eval/command.py → /Users/aayu/.codex/worktrees/66ca/Kahlus-V1/src/neurotwin/eval/paper_gate.py
- `Data schemas, split manifests, and leakage guards.` --uses--> `ArchitectureSpec`  [INFERRED]
  /Users/aayu/.codex/worktrees/66ca/Kahlus-V1/src/neurotwin/data/__init__.py → /Users/aayu/.codex/worktrees/66ca/Kahlus-V1/src/neurotwin/models/architecture_registry.py
- `Data schemas, split manifests, and leakage guards.` --uses--> `BaselineSpec`  [INFERRED]
  /Users/aayu/.codex/worktrees/66ca/Kahlus-V1/src/neurotwin/data/__init__.py → /Users/aayu/.codex/worktrees/66ca/Kahlus-V1/src/neurotwin/models/baselines.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.04
Nodes (93): ArchitectureSpec, ExecutableBaselineRunner, AggregatePayload, AggregateRankPayload, BaselineFailure, BaselineSuitePayload, BrainVistaStyleConfig, _causal_stimulus_features() (+85 more)

### Community 1 - "Community 1"
Cohesion: 0.04
Nodes (82): audit_prepared_eval_inputs(), audit_split_manifest(), AuditReport, _duplicate_metadata_value_violations(), _event_metadata_violations(), _forbidden_event_metadata_violations(), _forbidden_metadata(), _hidden_subject_metadata_violations() (+74 more)

### Community 2 - "Community 2"
Cohesion: 0.03
Nodes (58): BaseObservationOperator, Base class for latent-field-to-observation operators., BaseObservationOperator, BehaviorObservationOperator, Compile a latent neural field into behavior or task-label predictions., _expert_utilization(), NeuralFieldCompiler, NeuralFieldCompilerConfig (+50 more)

### Community 3 - "Community 3"
Cohesion: 0.05
Nodes (66): _mapping(), _optional_int(), _optional_nonnegative_float(), PreparedDataConfig, PreparedModelConfig, PreparedTrainingConfigInput, PreparedTrainingSectionConfig, _resolve_modalities() (+58 more)

### Community 4 - "Community 4"
Cohesion: 0.04
Nodes (50): _cmd_estimate(), _config_value(), _csv_cell(), _csv_rows(), _dry_run_result(), _finite_number(), _has_prepared_training_inputs(), _pair_operator_ablation_csv() (+42 more)

### Community 5 - "Community 5"
Cohesion: 0.06
Nodes (78): _aggregate_classification_inflation(), _aggregate_classification_interpretation(), _aggregate_classification_seed_results(), _aggregate_identity_risk(), _aggregate_identity_seed_results(), _aggregate_leakage_interpretation(), _aggregate_leakage_seed_results(), _bad_segment_split_classification_metrics() (+70 more)

### Community 6 - "Community 6"
Cohesion: 0.05
Nodes (30): architecture_registry(), architecture_spec(), architecture_status(), build_architecture_model(), estimate_architecture_extra_parameters(), _nfc_factory(), normalize_architecture_type(), _normalize_key() (+22 more)

### Community 7 - "Community 7"
Cohesion: 0.06
Nodes (52): _aggregate_rank_from_payload(), _aggregate_rank_matches(), aggregate_seed_metrics(), aggregate_seed_ranks(), aggregated_seed_tasks(), AggregateRankRecord, _audit_payload(), build_paper_mode_evidence() (+44 more)

### Community 8 - "Community 8"
Cohesion: 0.07
Nodes (32): _metrics(), bandpower_error(), bootstrap_ci(), mae(), mse(), pearsonr(), r2_score(), rank_models() (+24 more)

### Community 9 - "Community 9"
Cohesion: 0.08
Nodes (29): _bundle_rel_parts(), copy_bundle_file(), copy_current_docker_log(), copy_current_run_logs(), copy_source_file(), copy_tree_files(), current_docker_log_path(), current_slurm_job_id() (+21 more)

### Community 10 - "Community 10"
Cohesion: 0.07
Nodes (29): _add_eval_audit_args(), _add_eval_demo_args(), _add_eval_manifest_args(), _add_eval_suite_args(), _add_eval_window_args(), _cmd_data_audit(), _cmd_data_prepare(), _cmd_data_smoke() (+21 more)

### Community 11 - "Community 11"
Cohesion: 0.08
Nodes (34): _aggregate_seed_payloads(), _all_metrics_are_finite(), _criterion(), _falsification(), _fit_ridge(), _fit_sequence_baseline(), _format_falsification(), format_nfc_synthetic_report() (+26 more)

### Community 12 - "Community 12"
Cohesion: 0.09
Nodes (33): bids_manifest_summary(), _events_for(), _infer_modality(), _is_bids_signal(), _listlike(), _load_timeseries_derivative(), _parse_entities(), _read_tsv() (+25 more)

### Community 13 - "Community 13"
Cohesion: 0.07
Nodes (31): append_artifact_errors(), baseline_ranking_rows(), csv_cell(), csv_rows(), diagnostic_status(), first_json_artifact(), format_aggregate_rank(), is_artifact_error() (+23 more)

### Community 14 - "Community 14"
Cohesion: 0.11
Nodes (33): ensure_scripts_import_path(), ensure_src_import_path(), Allow lazy sibling script imports from module-based test loaders., Allow direct script execution without mutating imports at module load., main(), _flatten_metrics(), main(), effective_scientific_claim_allowed() (+25 more)

### Community 15 - "Community 15"
Cohesion: 0.11
Nodes (36): _dataset_site_generalization_from_windows(), _format_stimulus_evidence(), _group_windows(), run_prepared_auxiliary_tasks(), _scope_status(), _stimulus_evidence_from_tasks(), _subject_adaptation_from_windows(), _task_result_to_dict() (+28 more)

### Community 16 - "Community 16"
Cohesion: 0.12
Nodes (25): format_prepared_eval_audit(), _cmd_eval(), EvalCommandConfig, EvalCommandResult, _manifest_paths(), _paper_demo_config(), _paper_demo_error(), _paper_demo_exit_code() (+17 more)

### Community 17 - "Community 17"
Cohesion: 0.14
Nodes (13): _cmd_cluster_materialize_config(), _cmd_cluster_preflight(), _parse_split_windows(), ClusterMaterializeConfigReport, ClusterPreflightReport, format_cluster_materialize_config(), format_cluster_preflight(), materialize_cluster_config() (+5 more)

### Community 18 - "Community 18"
Cohesion: 0.13
Nodes (3): nfc_falsification_status(), ExpandedCliTests, _valid_paper_mode_gate()

### Community 19 - "Community 19"
Cohesion: 0.13
Nodes (1): ArtifactDocsContractsTests

### Community 20 - "Community 20"
Cohesion: 0.14
Nodes (6): _optional_float(), _optional_str(), preprocessing_hash(), sampling_rate(), source_hash(), split_assignment()

### Community 21 - "Community 21"
Cohesion: 0.19
Nodes (7): dataset_registry(), DatasetAdapterSpec, RegistryTests, permissive_upstreams(), quarantined_upstreams(), upstream_registry(), UpstreamSpec

### Community 22 - "Community 22"
Cohesion: 0.23
Nodes (4): assert_runner_archive(), copy_repo_to_temp_git(), HandoffZipArtifactTests, RunnerBundleArtifactTests

### Community 23 - "Community 23"
Cohesion: 0.25
Nodes (3): baseline_catalog_rows(), BaselineCatalogEntry, _baseline_catalog()

### Community 24 - "Community 24"
Cohesion: 0.53
Nodes (1): ManifestPersistenceAndEvalSuiteTests

### Community 25 - "Community 25"
Cohesion: 0.7
Nodes (4): main(), _nccl_version(), _payload(), _positive_int()

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (2): main(), _merge_dicts()

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (0):

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (0):

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (0):

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (0):

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (0):

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (0):

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (0):

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (0):

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (0):

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (0):

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (0):

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): Load a local NeuroTwin TRIBE-style config or seeded defaults.          No pretra

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Compatibility shim for TRIBE-style callers.          Prefer :meth:`from_checkpoi

## Knowledge Gaps
- **34 isolated node(s):** `Allow direct script execution without mutating imports at module load.`, `Allow lazy sibling script imports from module-based test loaders.`, `Narrow loaded YAML at the command boundary after load_config validation.`, `Raised when an experiment config cannot be loaded or validated.`, `Resolve a source commit from git, falling back to COMMIT_HASH.txt.` (+29 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 27`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `paper_mode.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Load a local NeuroTwin TRIBE-style config or seeded defaults.          No pretra`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Compatibility shim for TRIBE-style callers.          Prefer :meth:`from_checkpoi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `build_split_manifest()` connect `Community 1` to `Community 2`, `Community 4`, `Community 6`, `Community 8`, `Community 17`?**
  _High betweenness centrality (0.094) - this node is a cross-community bridge._
- **Why does `_require_bundle_rel()` connect `Community 9` to `Community 2`?**
  _High betweenness centrality (0.089) - this node is a cross-community bridge._
- **Why does `NeuralEventBatch` connect `Community 1` to `Community 0`, `Community 2`, `Community 3`, `Community 12`, `Community 20`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **Are the 87 inferred relationships involving `ValueError` (e.g. with `_require_bundle_rel()` and `resolve_prepared_config()`) actually correct?**
  _`ValueError` has 87 INFERRED edges - model-reasoned connections that need verification._
- **Are the 34 inferred relationships involving `NeuroTwinPairOperator` (e.g. with `ModelShapeTests` and `ArchitectureRegistryTests`) actually correct?**
  _`NeuroTwinPairOperator` has 34 INFERRED edges - model-reasoned connections that need verification._
- **Are the 39 inferred relationships involving `NeuralStateSpaceTranslator` (e.g. with `ModelMetadataGeometryTests` and `ModelShapeTests`) actually correct?**
  _`NeuralStateSpaceTranslator` has 39 INFERRED edges - model-reasoned connections that need verification._
- **Are the 37 inferred relationships involving `NeuralStateSpaceTranslatorConfig` (e.g. with `ModelMetadataGeometryTests` and `ModelShapeTests`) actually correct?**
  _`NeuralStateSpaceTranslatorConfig` has 37 INFERRED edges - model-reasoned connections that need verification._