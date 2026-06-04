# Graph Report - /Users/aayu/conductor/workspaces/kahlus-v1/belgrade-v1  (2026-06-03)

## Corpus Check
- 122 files · ~174,828 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1236 nodes · 3156 edges · 67 communities detected
- Extraction: 56% EXTRACTED · 44% INFERRED · 0% AMBIGUOUS · INFERRED: 1389 edges (avg confidence: 0.69)
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
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]

## God Nodes (most connected - your core abstractions)
1. `NeuralStateSpaceTranslator` - 53 edges
2. `NeuralStateSpaceTranslatorConfig` - 41 edges
3. `NumpyRidgeBaseline` - 39 edges
4. `NeuralEventBatch` - 39 edges
5. `build_split_manifest()` - 38 edges
6. `NeuroTwinPairOperator` - 37 edges
7. `SupervisedWindowTask` - 37 edges
8. `TorchMLPBaseline` - 33 edges
9. `TorchTCNBaseline` - 33 edges
10. `TinyTransformerBaseline` - 33 edges

## Surprising Connections (you probably didn't know these)
- `PaperModeEvidence` --uses--> `Validate the artifact contract required before paper-mode claims.      This is i`  [INFERRED]
  /Users/aayu/conductor/workspaces/kahlus-v1/belgrade-v1/src/neurotwin/eval/paper_contracts.py → src/neurotwin/eval/paper_gate.py
- `Data schemas, split manifests, and leakage guards.` --uses--> `TribeStyleModel`  [INFERRED]
  src/neurotwin/data/__init__.py → /Users/aayu/conductor/workspaces/kahlus-v1/belgrade-v1/src/neurotwin/models/tribe_style.py
- `Data schemas, split manifests, and leakage guards.` --uses--> `TribeStyleStimulusInput`  [INFERRED]
  src/neurotwin/data/__init__.py → /Users/aayu/conductor/workspaces/kahlus-v1/belgrade-v1/src/neurotwin/models/tribe_style.py
- `Data schemas, split manifests, and leakage guards.` --uses--> `RankingRow`  [INFERRED]
  src/neurotwin/data/__init__.py → /Users/aayu/conductor/workspaces/kahlus-v1/belgrade-v1/src/neurotwin/scoring/metrics.py
- `TaskResult` --uses--> `Run prepared baselines for multiple seeds and write paper-mode artifacts.`  [INFERRED]
  /Users/aayu/conductor/workspaces/kahlus-v1/belgrade-v1/src/neurotwin/benchmarks/tasks.py → src/neurotwin/benchmarks/prepared_suite.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.03
Nodes (121): audit_prepared_eval_inputs(), audit_split_manifest(), AuditReport, _forbidden_metadata(), _metadata_group_overlap(), PreparedEvalAuditReport, Audit prepared eval inputs before any benchmark score is trusted., Audit prepared eval inputs before any benchmark score is trusted. (+113 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (85): ExecutableBaselineRunner, AggregatePayload, AggregateRankPayload, BaselineFailure, BaselineSuitePayload, _causal_stimulus_features(), _cross_modal_task(), _fit_autoregressive_ridge() (+77 more)

### Community 2 - "Community 2"
Cohesion: 0.05
Nodes (69): _mapping(), _optional_int(), _optional_nonnegative_float(), PreparedDataConfig, PreparedModelConfig, PreparedTrainingConfigInput, PreparedTrainingSectionConfig, _resolve_modalities() (+61 more)

### Community 3 - "Community 3"
Cohesion: 0.04
Nodes (52): _cmd_estimate(), _cmd_train(), _config_value(), _csv_cell(), _csv_rows(), _dry_run_result(), _has_prepared_training_inputs(), _pair_operator_ablation_csv() (+44 more)

### Community 4 - "Community 4"
Cohesion: 0.05
Nodes (62): append_artifact_errors(), baseline_ranking_rows(), csv_cell(), csv_rows(), diagnostic_status(), first_json_artifact(), format_aggregate_rank(), is_artifact_error() (+54 more)

### Community 5 - "Community 5"
Cohesion: 0.07
Nodes (61): format_prepared_eval_audit(), _cmd_run(), EvalCommandConfig, EvalCommandResult, _manifest_paths(), _paper_demo_config(), _paper_demo_error(), _paper_demo_exit_code() (+53 more)

### Community 6 - "Community 6"
Cohesion: 0.07
Nodes (31): _metrics(), bandpower_error(), bootstrap_ci(), mae(), mse(), pearsonr(), r2_score(), _rankdata() (+23 more)

### Community 7 - "Community 7"
Cohesion: 0.08
Nodes (29): _bundle_rel_parts(), copy_bundle_file(), copy_current_docker_log(), copy_current_run_logs(), copy_source_file(), copy_tree_files(), current_docker_log_path(), current_slurm_job_id() (+21 more)

### Community 8 - "Community 8"
Cohesion: 0.08
Nodes (47): rank_models(), _aggregate_rank_from_payload(), _aggregate_rank_matches(), aggregate_seed_metrics(), aggregate_seed_ranks(), aggregated_seed_tasks(), AggregateRankRecord, _audit_payload() (+39 more)

### Community 9 - "Community 9"
Cohesion: 0.06
Nodes (19): _build_moabb_paradigm(), _resolve_moabb_dataset(), _build_temporal_core(), _build_value_projector(), _check_sequence_tensor(), from_mapping(), _pair_key(), _resolve_pair_operator_config() (+11 more)

### Community 10 - "Community 10"
Cohesion: 0.08
Nodes (20): baseline_catalog_rows(), BaselineCatalogEntry, _baseline_catalog(), run_supervised_window_tasks(), BaselineSuiteTests, _build_events(), _event_embedding(), from_checkpoint() (+12 more)

### Community 11 - "Community 11"
Cohesion: 0.09
Nodes (21): _add_eval_audit_args(), _add_eval_demo_args(), _add_eval_manifest_args(), _add_eval_suite_args(), _add_eval_window_args(), _cmd_doctor(), main(), _normalize_eval_argv() (+13 more)

### Community 12 - "Community 12"
Cohesion: 0.11
Nodes (14): _baseline_csv_has_rankings(), _baseline_ranking_present(), build_prepared_evidence_gate(), _competitor_reproduction_status_present(), format_evidence_diagnostic_report(), _paper_mode_gate_present(), _prepared_suite_has_rankings(), read_json_artifact() (+6 more)

### Community 13 - "Community 13"
Cohesion: 0.14
Nodes (13): _cmd_cluster_materialize_config(), _cmd_cluster_preflight(), _parse_split_windows(), ClusterMaterializeConfigReport, ClusterPreflightReport, format_cluster_materialize_config(), format_cluster_preflight(), materialize_cluster_config() (+5 more)

### Community 14 - "Community 14"
Cohesion: 0.16
Nodes (26): build_future_forecasting_task_from_windows(), build_prepared_window_tasks(), _cross_modal_task_from_windows(), _future_task_from_windows(), _future_xy(), _masked_task_from_windows(), _metadata_list(), _normalize_hash() (+18 more)

### Community 15 - "Community 15"
Cohesion: 0.18
Nodes (2): ExpandedCliTests, _valid_paper_mode_gate()

### Community 16 - "Community 16"
Cohesion: 0.21
Nodes (16): _events_for(), _infer_modality(), _is_bids_signal(), _listlike(), _load_timeseries_derivative(), _parse_entities(), _read_tsv(), _read_tsv_rows() (+8 more)

### Community 17 - "Community 17"
Cohesion: 0.13
Nodes (1): ArtifactDocsContractsTests

### Community 18 - "Community 18"
Cohesion: 0.14
Nodes (6): _optional_float(), _optional_str(), preprocessing_hash(), sampling_rate(), source_hash(), split_assignment()

### Community 19 - "Community 19"
Cohesion: 0.19
Nodes (7): dataset_registry(), DatasetAdapterSpec, RegistryTests, permissive_upstreams(), quarantined_upstreams(), upstream_registry(), UpstreamSpec

### Community 20 - "Community 20"
Cohesion: 0.24
Nodes (4): assert_runner_archive(), copy_repo_to_temp_git(), HandoffZipArtifactTests, RunnerBundleArtifactTests

### Community 21 - "Community 21"
Cohesion: 0.44
Nodes (9): _duplicate_metadata_value_violations(), _event_metadata_violations(), _forbidden_event_metadata_violations(), _hidden_subject_metadata_violations(), _prepared_windows_by_split(), _record_id(), _stimulus_segment_key(), _stimulus_segment_violations() (+1 more)

### Community 22 - "Community 22"
Cohesion: 0.7
Nodes (4): main(), _nccl_version(), _payload(), _positive_int()

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (0):

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (0):

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (0):

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (0):

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
Nodes (1): Load a local NeuroTwin TRIBE-style config or seeded defaults.          No pretra

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): Compatibility shim for TRIBE-style callers.          Prefer :meth:`from_checkpoi

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (1): Narrow loaded YAML at the command boundary after load_config validation.

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): Narrow loaded YAML at the command boundary after load_config validation.

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): Compatibility shim for report helpers now owned by neurotwin.benchmarks.  TODO:

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Validate the artifact contract required before paper-mode claims.      This is i

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Resolve a source commit from git, falling back to COMMIT_HASH.txt.

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): Validate cluster launch inputs before an expensive SLURM allocation runs.

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): Write a cluster config with absolute prepared-manifest paths.

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): NeuroTwin-native stimulus-to-fMRI encoder for the TRIBE-style lane.

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): Small TRIBE-compatible facade implemented entirely with NeuroTwin code.      Thi

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): Load a local NeuroTwin TRIBE-style config or seeded defaults.          No pretra

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Compatibility shim for TRIBE-style callers.          Prefer :meth:`from_checkpoi

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): Build minimal local event rows for smoke/pipeline tests.          Text events ar

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): Compatibility shim returning local event rows, not a pandas DataFrame.

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Predict toy fMRI responses from event rows.          Event rows are converted to

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Configuration shell for the future NeuroTwin model implementation.

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Small Transformer baseline for CPU shape and smoke tests.

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): CPU debug stand-in for long-sequence SSM baselines until Mamba is pinned.

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): Modality encoders + shared latent dynamics + modality readouts.

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): Encode observed modalities into shared latent neural tokens.

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): Validate the artifact contract required before paper-mode claims.      This is i

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): Validate the artifact contract required before paper-mode claims.      This is i

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): Validate the artifact contract required before paper-mode claims.      This is i

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (1): Validate the artifact contract required before paper-mode claims.      This is i

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (1): Validate cluster launch inputs before an expensive SLURM allocation runs.

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (1): Per-timepoint MLP baseline for neural windows.

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (1): Small Conv1D/TCN-style baseline with stable [batch, time, feature] IO.

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (1): Small Transformer baseline for CPU shape and smoke tests.

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (1): CPU debug stand-in for long-sequence SSM baselines until Mamba is pinned.

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (1): Modality-tokenizer + shared latent dynamics + modality-readout scaffold.

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (1): Encode observed modalities into shared latent neural tokens.

### Community 66 - "Community 66"
Cohesion: 1.0
Nodes (1): Encode observed modalities into shared latent neural tokens.

## Knowledge Gaps
- **62 isolated node(s):** `Allow direct script execution without mutating imports at module load.`, `Allow lazy sibling script imports from module-based test loaders.`, `Narrow loaded YAML at the command boundary after load_config validation.`, `Raised when an experiment config cannot be loaded or validated.`, `Resolve a source commit from git, falling back to COMMIT_HASH.txt.` (+57 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 23`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
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
- **Thin community `Community 33`** (1 nodes): `paper_mode.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `Load a local NeuroTwin TRIBE-style config or seeded defaults.          No pretra`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `Compatibility shim for TRIBE-style callers.          Prefer :meth:`from_checkpoi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `Narrow loaded YAML at the command boundary after load_config validation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `Narrow loaded YAML at the command boundary after load_config validation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Compatibility shim for report helpers now owned by neurotwin.benchmarks.  TODO:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Validate the artifact contract required before paper-mode claims.      This is i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Resolve a source commit from git, falling back to COMMIT_HASH.txt.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Validate cluster launch inputs before an expensive SLURM allocation runs.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Write a cluster config with absolute prepared-manifest paths.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `NeuroTwin-native stimulus-to-fMRI encoder for the TRIBE-style lane.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `Small TRIBE-compatible facade implemented entirely with NeuroTwin code.      Thi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Load a local NeuroTwin TRIBE-style config or seeded defaults.          No pretra`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Compatibility shim for TRIBE-style callers.          Prefer :meth:`from_checkpoi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `Build minimal local event rows for smoke/pipeline tests.          Text events ar`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `Compatibility shim returning local event rows, not a pandas DataFrame.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Predict toy fMRI responses from event rows.          Event rows are converted to`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Configuration shell for the future NeuroTwin model implementation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Small Transformer baseline for CPU shape and smoke tests.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `CPU debug stand-in for long-sequence SSM baselines until Mamba is pinned.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Modality encoders + shared latent dynamics + modality readouts.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `Encode observed modalities into shared latent neural tokens.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `Validate the artifact contract required before paper-mode claims.      This is i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `Validate the artifact contract required before paper-mode claims.      This is i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `Validate the artifact contract required before paper-mode claims.      This is i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `Validate the artifact contract required before paper-mode claims.      This is i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (1 nodes): `Validate cluster launch inputs before an expensive SLURM allocation runs.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (1 nodes): `Per-timepoint MLP baseline for neural windows.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (1 nodes): `Small Conv1D/TCN-style baseline with stable [batch, time, feature] IO.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (1 nodes): `Small Transformer baseline for CPU shape and smoke tests.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (1 nodes): `CPU debug stand-in for long-sequence SSM baselines until Mamba is pinned.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (1 nodes): `Modality-tokenizer + shared latent dynamics + modality-readout scaffold.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (1 nodes): `Encode observed modalities into shared latent neural tokens.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 66`** (1 nodes): `Encode observed modalities into shared latent neural tokens.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_cmd_train()` connect `Community 3` to `Community 0`, `Community 1`, `Community 11`, `Community 12`?**
  _High betweenness centrality (0.084) - this node is a cross-community bridge._
- **Why does `run_prepared_training()` connect `Community 12` to `Community 0`, `Community 1`, `Community 2`, `Community 3`, `Community 6`, `Community 9`?**
  _High betweenness centrality (0.075) - this node is a cross-community bridge._
- **Why does `_require_bundle_rel()` connect `Community 7` to `Community 9`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Are the 64 inferred relationships involving `ValueError` (e.g. with `_require_bundle_rel()` and `resolve_prepared_config()`) actually correct?**
  _`ValueError` has 64 INFERRED edges - model-reasoned connections that need verification._
- **Are the 46 inferred relationships involving `NeuralStateSpaceTranslator` (e.g. with `ModelMetadataGeometryTests` and `ModelShapeTests`) actually correct?**
  _`NeuralStateSpaceTranslator` has 46 INFERRED edges - model-reasoned connections that need verification._
- **Are the 38 inferred relationships involving `NeuralStateSpaceTranslatorConfig` (e.g. with `ModelMetadataGeometryTests` and `ModelShapeTests`) actually correct?**
  _`NeuralStateSpaceTranslatorConfig` has 38 INFERRED edges - model-reasoned connections that need verification._
- **Are the 34 inferred relationships involving `NumpyRidgeBaseline` (e.g. with `BaselinesAndArchitectureTests` and `BaselineSuiteTests`) actually correct?**
  _`NumpyRidgeBaseline` has 34 INFERRED edges - model-reasoned connections that need verification._