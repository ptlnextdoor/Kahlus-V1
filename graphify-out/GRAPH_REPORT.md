# Graph Report - /Users/aayu/conductor/workspaces/kahlus-v1/little-rock  (2026-06-01)

## Corpus Check
- 84 files · ~113,827 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 805 nodes · 1828 edges · 36 communities detected
- Extraction: 62% EXTRACTED · 38% INFERRED · 0% AMBIGUOUS · INFERRED: 687 edges (avg confidence: 0.74)
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

## God Nodes (most connected - your core abstractions)
1. `NeuralEventBatch` - 35 edges
2. `run_prepared_training()` - 30 edges
3. `NeuralStateSpaceTranslator` - 30 edges
4. `build_split_manifest()` - 29 edges
5. `audit_prepared_eval_inputs()` - 26 edges
6. `_cmd_train()` - 24 edges
7. `make_synthetic_recordings()` - 22 edges
8. `RecordingRecord` - 22 edges
9. `EvidenceBundleArtifactTests` - 21 edges
10. `SupervisedWindowTask` - 21 edges

## Surprising Connections (you probably didn't know these)
- `Run a tiny deterministic CPU training loop for CLI and CI smoke tests.` --uses--> `NeuralStateSpaceTranslator`  [INFERRED]
  src/neurotwin/training/smoke.py → /Users/aayu/conductor/workspaces/kahlus-v1/little-rock/src/neurotwin/models/torch_models.py
- `TribeStyleStimulusEncoder` --uses--> `Run tiny local baselines on paired synthetic windows.      This is intentionally`  [INFERRED]
  /Users/aayu/conductor/workspaces/kahlus-v1/little-rock/src/neurotwin/models/tribe_style.py → src/neurotwin/benchmarks/baseline_suite.py
- `Load a small MOABB dataset through its paradigm API when optional deps exist.` --uses--> `NeuralEventBatch`  [INFERRED]
  src/neurotwin/adapters/moabb.py → /Users/aayu/conductor/workspaces/kahlus-v1/little-rock/src/neurotwin/data/schemas.py
- `Load a small MOABB dataset through its paradigm API when optional deps exist.` --uses--> `RecordingRecord`  [INFERRED]
  src/neurotwin/adapters/moabb.py → /Users/aayu/conductor/workspaces/kahlus-v1/little-rock/src/neurotwin/data/split_manifest.py
- `Cap MOABB smoke trials without collapsing a group-held-out split.` --uses--> `NeuralEventBatch`  [INFERRED]
  src/neurotwin/adapters/moabb.py → /Users/aayu/conductor/workspaces/kahlus-v1/little-rock/src/neurotwin/data/schemas.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.04
Nodes (71): audit_prepared_eval_inputs(), audit_split_manifest(), AuditReport, _duplicate_metadata_value_violations(), _event_metadata_violations(), _forbidden_event_metadata_violations(), _forbidden_metadata(), _hidden_subject_metadata_violations() (+63 more)

### Community 1 - "Community 1"
Cohesion: 0.04
Nodes (59): _baseline_catalog(), BaselineFailure, _cross_modal_task(), _fit_autoregressive_ridge(), _fit_neurotwin(), _fit_ridge(), _fit_torch_sequence_model(), _fit_tribe_style() (+51 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (48): bids_manifest_summary(), _events_for(), _infer_modality(), _is_bids_signal(), _listlike(), _load_timeseries_derivative(), _parse_entities(), _read_tsv() (+40 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (40): _cmd_cluster_preflight(), _cmd_doctor(), _cmd_estimate(), _cmd_split_audit(), _cmd_train(), _config_value(), _has_prepared_training_inputs(), _parse_split_windows() (+32 more)

### Community 4 - "Community 4"
Cohesion: 0.07
Nodes (28): cleanup_process_group(), DistributedInfo, get_distributed_info(), get_rank_metrics_path(), maybe_init_process_group(), unwrap_model(), wrap_ddp_if_initialized(), _aggregate_task_results() (+20 more)

### Community 5 - "Community 5"
Cohesion: 0.08
Nodes (40): format_prepared_eval_audit(), _cmd_eval(), EvalCommandResult, _manifest_paths(), _prepared_suite_config(), _run_audit_command(), run_eval_command(), _run_neural_translation_v1_command() (+32 more)

### Community 6 - "Community 6"
Cohesion: 0.09
Nodes (27): _bundle_rel_parts(), copy_bundle_file(), copy_current_docker_log(), copy_current_run_logs(), copy_source_file(), copy_tree_files(), current_docker_log_path(), current_slurm_job_id() (+19 more)

### Community 7 - "Community 7"
Cohesion: 0.08
Nodes (45): _cmd_report(), main(), _flatten_metrics(), main(), _aggregate_rank(), _aggregate_rank_from_seed_records(), _aggregate_rank_matches(), _audit_payload() (+37 more)

### Community 8 - "Community 8"
Cohesion: 0.08
Nodes (28): _metrics(), bandpower_error(), bootstrap_ci(), mae(), mse(), pearsonr(), r2_score(), rank_models() (+20 more)

### Community 9 - "Community 9"
Cohesion: 0.15
Nodes (12): _cmd_cluster_materialize_config(), ClusterMaterializeConfigReport, ClusterPreflightReport, _config_value(), format_cluster_materialize_config(), format_cluster_preflight(), materialize_cluster_config(), Write a cluster config with absolute prepared-manifest paths. (+4 more)

### Community 10 - "Community 10"
Cohesion: 0.1
Nodes (12): Data schemas, split manifests, and leakage guards., _build_events(), _event_embedding(), from_checkpoint(), from_pretrained(), _load_local_config(), Build minimal local event rows for smoke/pipeline tests.          Text events ar, Compatibility shim returning local event rows, not a pandas DataFrame. (+4 more)

### Community 11 - "Community 11"
Cohesion: 0.22
Nodes (18): build_prepared_window_tasks(), _cross_modal_task_from_windows(), _dataset_site_generalization_from_windows(), _first_modality_with_splits(), _future_task_from_windows(), _future_xy(), _group_windows(), _masked_task_from_windows() (+10 more)

### Community 12 - "Community 12"
Cohesion: 0.13
Nodes (1): ArtifactDocsContractsTests

### Community 13 - "Community 13"
Cohesion: 0.15
Nodes (6): _optional_float(), _optional_str(), preprocessing_hash(), sampling_rate(), source_hash(), split_assignment()

### Community 14 - "Community 14"
Cohesion: 0.23
Nodes (2): ExpandedCliTests, _valid_paper_mode_gate()

### Community 15 - "Community 15"
Cohesion: 0.2
Nodes (7): dataset_registry(), DatasetAdapterSpec, RegistryTests, permissive_upstreams(), quarantined_upstreams(), upstream_registry(), UpstreamSpec

### Community 16 - "Community 16"
Cohesion: 0.24
Nodes (4): assert_runner_archive(), copy_repo_to_temp_git(), HandoffZipArtifactTests, RunnerBundleArtifactTests

### Community 17 - "Community 17"
Cohesion: 0.7
Nodes (4): main(), _nccl_version(), _payload(), _positive_int()

### Community 18 - "Community 18"
Cohesion: 0.5
Nodes (2): NeuralStateSpaceTranslatorConfig, Configuration shell for the future NeuroTwin model implementation.

### Community 19 - "Community 19"
Cohesion: 0.67
Nodes (1): CliReportTests

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (2): _clone_or_checkout(), main()

### Community 21 - "Community 21"
Cohesion: 1.0
Nodes (0): 

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (1): Load a local NeuroTwin TRIBE-style config or seeded defaults.          No pretra

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (1): Compatibility shim for TRIBE-style callers.          Prefer :meth:`from_checkpoi

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (1): Validate the artifact contract required before paper-mode claims.      This is i

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): Validate the artifact contract required before paper-mode claims.      This is i

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): Validate the artifact contract required before paper-mode claims.      This is i

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): Validate the artifact contract required before paper-mode claims.      This is i

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (1): Validate cluster launch inputs before an expensive SLURM allocation runs.

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): Per-timepoint MLP baseline for neural windows.

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): Small Conv1D/TCN-style baseline with stable [batch, time, feature] IO.

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): Small Transformer baseline for CPU shape and smoke tests.

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): CPU debug stand-in for long-sequence SSM baselines until Mamba is pinned.

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (1): Modality-tokenizer + shared latent dynamics + modality-readout scaffold.

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): Encode observed modalities into shared latent neural tokens.

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): Encode observed modalities into shared latent neural tokens.

## Knowledge Gaps
- **37 isolated node(s):** `Raised when an experiment config cannot be loaded or validated.`, `Resolve a source commit from git, falling back to COMMIT_HASH.txt.`, `Validate cluster launch inputs before an expensive SLURM allocation runs.`, `Write a cluster config with absolute prepared-manifest paths.`, `Closed-form ridge baseline for sanity checks and tiny CPU benchmarks.` (+32 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 21`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `Load a local NeuroTwin TRIBE-style config or seeded defaults.          No pretra`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `Compatibility shim for TRIBE-style callers.          Prefer :meth:`from_checkpoi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `Validate the artifact contract required before paper-mode claims.      This is i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `Validate the artifact contract required before paper-mode claims.      This is i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `Validate the artifact contract required before paper-mode claims.      This is i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `Validate the artifact contract required before paper-mode claims.      This is i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `Validate cluster launch inputs before an expensive SLURM allocation runs.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `Per-timepoint MLP baseline for neural windows.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `Small Conv1D/TCN-style baseline with stable [batch, time, feature] IO.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Small Transformer baseline for CPU shape and smoke tests.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `CPU debug stand-in for long-sequence SSM baselines until Mamba is pinned.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `Modality-tokenizer + shared latent dynamics + modality-readout scaffold.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `Encode observed modalities into shared latent neural tokens.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `Encode observed modalities into shared latent neural tokens.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_require_bundle_rel()` connect `Community 6` to `Community 1`?**
  _High betweenness centrality (0.107) - this node is a cross-community bridge._
- **Why does `run_prepared_training()` connect `Community 4` to `Community 0`, `Community 1`, `Community 2`, `Community 3`, `Community 5`, `Community 8`, `Community 11`?**
  _High betweenness centrality (0.104) - this node is a cross-community bridge._
- **Why does `_cmd_train()` connect `Community 3` to `Community 0`, `Community 1`, `Community 2`, `Community 4`, `Community 5`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **Are the 43 inferred relationships involving `ValueError` (e.g. with `_require_bundle_rel()` and `generate_suite_report()`) actually correct?**
  _`ValueError` has 43 INFERRED edges - model-reasoned connections that need verification._
- **Are the 32 inferred relationships involving `NeuralEventBatch` (e.g. with `NeuralEventBatchTests` and `Create paired synthetic recordings that mirror the recording manifest.`) actually correct?**
  _`NeuralEventBatch` has 32 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `run_prepared_training()` (e.g. with `.test_prepared_training_writes_checkpoint_and_metrics()` and `.test_prepared_training_runs_all_neural_translation_tasks()`) actually correct?**
  _`run_prepared_training()` has 20 INFERRED edges - model-reasoned connections that need verification._
- **Are the 23 inferred relationships involving `NeuralStateSpaceTranslator` (e.g. with `ModelMetadataGeometryTests` and `BaselinesAndArchitectureTests`) actually correct?**
  _`NeuralStateSpaceTranslator` has 23 INFERRED edges - model-reasoned connections that need verification._