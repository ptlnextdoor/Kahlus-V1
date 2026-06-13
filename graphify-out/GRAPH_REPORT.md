# Graph Report - /Users/aayu/conductor/workspaces/kahlus-v2/irvine  (2026-06-13)

## Corpus Check
- 201 files · ~269,095 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1887 nodes · 4912 edges · 44 communities detected
- Extraction: 56% EXTRACTED · 44% INFERRED · 0% AMBIGUOUS · INFERRED: 2177 edges (avg confidence: 0.71)
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

## God Nodes (most connected - your core abstractions)
1. `Data schemas, split manifests, and leakage guards.` - 59 edges
2. `NumpyRidgeBaseline` - 58 edges
3. `NeuralStateSpaceTranslator` - 50 edges
4. `NeuroTwinPairOperator` - 48 edges
5. `NeuralStateSpaceTranslatorConfig` - 44 edges
6. `DualFieldConfig` - 41 edges
7. `TorchMLPBaseline` - 40 edges
8. `TinyTransformerBaseline` - 40 edges
9. `TinySSMBaseline` - 40 edges
10. `build_split_manifest()` - 40 edges

## Surprising Connections (you probably didn't know these)
- `run_prepared_eval_command()` --calls--> `format_paper_mode_gate()`  [INFERRED]
  /Users/aayu/conductor/workspaces/kahlus-v2/irvine/src/neurotwin/eval/command.py → /Users/aayu/conductor/workspaces/kahlus-v2/irvine/src/neurotwin/eval/paper_gate.py
- `Forecast quality for flattened trajectory predictions.` --uses--> `PerturbationLibrary`  [INFERRED]
  /Users/aayu/conductor/workspaces/kahlus-v2/irvine/src/neurotwin/transition_gym/metrics.py → /Users/aayu/conductor/workspaces/kahlus-v2/irvine/src/neurotwin/transition_gym/perturbation_library.py
- `Data schemas, split manifests, and leakage guards.` --uses--> `EMContext`  [INFERRED]
  /Users/aayu/.codex/worktrees/66ca/Kahlus-V1/src/neurotwin/data/__init__.py → /Users/aayu/conductor/workspaces/kahlus-v2/irvine/src/neurotwin/em/em_context_schema.py
- `Data schemas, split manifests, and leakage guards.` --uses--> `IdleRecordingMetadata`  [INFERRED]
  /Users/aayu/.codex/worktrees/66ca/Kahlus-V1/src/neurotwin/data/__init__.py → /Users/aayu/conductor/workspaces/kahlus-v2/irvine/src/neurotwin/em/em_context_schema.py
- `Data schemas, split manifests, and leakage guards.` --uses--> `PhantomRecordingSchema`  [INFERRED]
  /Users/aayu/.codex/worktrees/66ca/Kahlus-V1/src/neurotwin/data/__init__.py → /Users/aayu/conductor/workspaces/kahlus-v2/irvine/src/neurotwin/em/em_context_schema.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.04
Nodes (109): ArchitectureSpec, BaselineRunResult, _fit_ridge(), _flatten_window(), _last_step(), _predict_torch(), Shared local baseline runner for the v2 dual-field and v3 Transition Gym tasks., Return a prediction array for ``model_id`` on ``task.x_test`` or raise on skip. (+101 more)

### Community 1 - "Community 1"
Cohesion: 0.03
Nodes (95): audit_prepared_eval_inputs(), audit_split_manifest(), AuditReport, _duplicate_metadata_value_violations(), _event_metadata_violations(), _forbidden_event_metadata_violations(), _forbidden_metadata(), _hidden_subject_metadata_violations() (+87 more)

### Community 2 - "Community 2"
Cohesion: 0.03
Nodes (142): ensure_src_import_path(), Allow direct script execution without mutating imports at module load., main(), _nccl_version(), _payload(), _positive_int(), _collect_metrics(), evaluate_gate() (+134 more)

### Community 3 - "Community 3"
Cohesion: 0.04
Nodes (62): _cmd_train(), _config_value(), _csv_cell(), _csv_rows(), _dry_run_result(), _finite_number(), _has_prepared_training_inputs(), _pair_operator_ablation_csv() (+54 more)

### Community 4 - "Community 4"
Cohesion: 0.04
Nodes (75): dual_field_regression_task(), run_baselines(), _all_finite(), benchmark_report(), v2 dual-field synthetic falsification benchmark.  Runs the diagnostic battery, t, run_v2_benchmark(), V2BenchmarkResult, write_v2_report() (+67 more)

### Community 5 - "Community 5"
Cohesion: 0.05
Nodes (69): _mapping(), _optional_int(), _optional_nonnegative_float(), PreparedDataConfig, PreparedModelConfig, PreparedTrainingConfigInput, PreparedTrainingSectionConfig, _resolve_modalities() (+61 more)

### Community 6 - "Community 6"
Cohesion: 0.04
Nodes (64): AlgonautsPrepareResult, _align_response_and_stimulus(), _cached_hash(), _candidate_feature_files(), _candidate_response_files(), _canonical_stimulus_id(), _compatible_time_length(), _discover_response_records() (+56 more)

### Community 7 - "Community 7"
Cohesion: 0.03
Nodes (32): BaseObservationOperator, Base class for latent-field-to-observation operators., BaseObservationOperator, BehaviorObservationOperator, Compile a latent neural field into behavior or task-label predictions., _expert_utilization(), from_mapping(), NeuralFieldCompiler (+24 more)

### Community 8 - "Community 8"
Cohesion: 0.05
Nodes (41): KTMConfig, Configuration for the Kahlus v2 synthetic dual-field scaffold.  PROPOSED / SYNTH, HistoryEncoder, History encoder: maps a window of EEG-like observations to an embedding h_t., KTM, KTM (Kahlus Transition Model) orchestrator scaffold.  Wires the history encoder,, C_K(h_t): ``(batch, K, horizon, eeg_channels)``., pθ(τ | h_t, a_k): ``(batch, horizon, eeg_channels)``. (+33 more)

### Community 9 - "Community 9"
Cohesion: 0.04
Nodes (50): format_artifact_report_md(), Stage 0 artifact audit: does environment/device change affect EEG hardware (no b, Synthesize a phantom/idle EEG-like recording ``(n_channels, n_samples)``.      B, Compute artifact features for both conditions and summarize the descriptive resp, run_artifact_audit(), synthesize_idle_recording(), band_power(), channel_artifact_features() (+42 more)

### Community 10 - "Community 10"
Cohesion: 0.05
Nodes (41): _add_eval_audit_args(), _add_eval_demo_args(), _add_eval_manifest_args(), _add_eval_suite_args(), _add_eval_window_args(), _cmd_cluster_materialize_config(), _cmd_cluster_preflight(), _cmd_data_audit() (+33 more)

### Community 11 - "Community 11"
Cohesion: 0.06
Nodes (52): _aggregate_rank_from_payload(), _aggregate_rank_matches(), aggregate_seed_metrics(), aggregate_seed_ranks(), aggregated_seed_tasks(), AggregateRankRecord, _audit_payload(), build_paper_mode_evidence() (+44 more)

### Community 12 - "Community 12"
Cohesion: 0.05
Nodes (35): transition_gym_regression_task(), build_data_card(), Data card generation for a Transition Gym instance.  The data card is a machine-, build_transition_gym(), TransitionGymBundle, mean_commutator_gap(), Compatibility re-export for legacy neurotwin.eval.metrics imports.  New code sho, Mean AB-vs-BA gap over a set of ordered perturbation pairs.      A value clearly (+27 more)

### Community 13 - "Community 13"
Cohesion: 0.08
Nodes (31): ensure_scripts_import_path(), Allow lazy sibling script imports from module-based test loaders., _bundle_rel_parts(), copy_bundle_file(), copy_current_docker_log(), copy_current_run_logs(), copy_source_file(), copy_tree_files() (+23 more)

### Community 14 - "Community 14"
Cohesion: 0.07
Nodes (34): _metrics(), _metrics(), bandpower_error(), bootstrap_ci(), mae(), mse(), pearsonr(), r2_score() (+26 more)

### Community 15 - "Community 15"
Cohesion: 0.07
Nodes (44): _aggregate_seed_payloads(), _all_metrics_are_finite(), _criterion(), _evidence_gate(), _falsification(), _fit_current_neurotwin(), _fit_nfc(), _fit_pair_operator() (+36 more)

### Community 16 - "Community 16"
Cohesion: 0.07
Nodes (31): append_artifact_errors(), baseline_ranking_rows(), csv_cell(), csv_rows(), diagnostic_status(), first_json_artifact(), format_aggregate_rank(), is_artifact_error() (+23 more)

### Community 17 - "Community 17"
Cohesion: 0.11
Nodes (36): _dataset_site_generalization_from_windows(), _format_stimulus_evidence(), _group_windows(), run_prepared_auxiliary_tasks(), _scope_status(), _stimulus_evidence_from_tasks(), _subject_adaptation_from_windows(), _task_result_to_dict() (+28 more)

### Community 18 - "Community 18"
Cohesion: 0.09
Nodes (17): architecture_registry(), architecture_spec(), architecture_status(), build_architecture_model(), estimate_architecture_extra_parameters(), _nfc_factory(), normalize_architecture_type(), _normalize_key() (+9 more)

### Community 19 - "Community 19"
Cohesion: 0.12
Nodes (25): format_prepared_eval_audit(), _cmd_eval(), EvalCommandConfig, EvalCommandResult, _manifest_paths(), _paper_demo_config(), _paper_demo_error(), _paper_demo_exit_code() (+17 more)

### Community 20 - "Community 20"
Cohesion: 0.13
Nodes (21): bids_manifest_summary(), _events_for(), _infer_modality(), _is_bids_signal(), _listlike(), _load_timeseries_derivative(), _parse_entities(), _read_tsv() (+13 more)

### Community 21 - "Community 21"
Cohesion: 0.15
Nodes (2): ExpandedCliTests, _valid_paper_mode_gate()

### Community 22 - "Community 22"
Cohesion: 0.13
Nodes (1): ArtifactDocsContractsTests

### Community 23 - "Community 23"
Cohesion: 0.19
Nodes (7): dataset_registry(), DatasetAdapterSpec, RegistryTests, permissive_upstreams(), quarantined_upstreams(), upstream_registry(), UpstreamSpec

### Community 24 - "Community 24"
Cohesion: 0.23
Nodes (4): assert_runner_archive(), copy_repo_to_temp_git(), HandoffZipArtifactTests, RunnerBundleArtifactTests

### Community 25 - "Community 25"
Cohesion: 0.25
Nodes (3): baseline_catalog_rows(), BaselineCatalogEntry, _baseline_catalog()

### Community 26 - "Community 26"
Cohesion: 0.46
Nodes (7): evaluate_debug_gate(), _finite_number(), _float_or_nan(), main(), _nonfinite_number_paths(), _read_json(), _task_result()

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
Nodes (0): 

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (0): 

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (0): 

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): Load a local NeuroTwin TRIBE-style config or seeded defaults.          No pretra

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): Compatibility shim for TRIBE-style callers.          Prefer :meth:`from_checkpoi

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): Narrow loaded YAML at the command boundary after load_config validation.

## Knowledge Gaps
- **101 isolated node(s):** `Allow direct script execution without mutating imports at module load.`, `Allow lazy sibling script imports from module-based test loaders.`, `Narrow loaded YAML at the command boundary after load_config validation.`, `Raised when an experiment config cannot be loaded or validated.`, `Resolve a source commit from git, falling back to COMMIT_HASH.txt.` (+96 more)
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
- **Thin community `Community 37`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `paper_mode.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Load a local NeuroTwin TRIBE-style config or seeded defaults.          No pretra`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Compatibility shim for TRIBE-style callers.          Prefer :meth:`from_checkpoi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Narrow loaded YAML at the command boundary after load_config validation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Data schemas, split manifests, and leakage guards.` connect `Community 7` to `Community 0`, `Community 4`, `Community 6`, `Community 8`, `Community 9`, `Community 12`, `Community 14`?**
  _High betweenness centrality (0.087) - this node is a cross-community bridge._
- **Why does `PerturbationLibrary` connect `Community 12` to `Community 2`, `Community 14`, `Community 7`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Why does `write_json()` connect `Community 3` to `Community 0`, `Community 1`, `Community 2`, `Community 4`, `Community 6`, `Community 8`, `Community 9`, `Community 11`, `Community 12`, `Community 14`, `Community 16`, `Community 19`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._
- **Are the 114 inferred relationships involving `ValueError` (e.g. with `_require_bundle_rel()` and `resolve_prepared_config()`) actually correct?**
  _`ValueError` has 114 INFERRED edges - model-reasoned connections that need verification._
- **Are the 41 inferred relationships involving `Data schemas, split manifests, and leakage guards.` (e.g. with `EMContext` and `IdleRecordingMetadata`) actually correct?**
  _`Data schemas, split manifests, and leakage guards.` has 41 INFERRED edges - model-reasoned connections that need verification._
- **Are the 53 inferred relationships involving `NumpyRidgeBaseline` (e.g. with `DualFieldTests` and `BaselinesAndArchitectureTests`) actually correct?**
  _`NumpyRidgeBaseline` has 53 INFERRED edges - model-reasoned connections that need verification._
- **Are the 43 inferred relationships involving `NeuralStateSpaceTranslator` (e.g. with `ModelMetadataGeometryTests` and `ModelShapeTests`) actually correct?**
  _`NeuralStateSpaceTranslator` has 43 INFERRED edges - model-reasoned connections that need verification._