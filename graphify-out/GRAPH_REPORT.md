# Graph Report - /Users/aayu/conductor/workspaces/kahlus-v2/irvine  (2026-06-13)

## Corpus Check
- 206 files · ~280,436 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1977 nodes · 5150 edges · 52 communities detected
- Extraction: 55% EXTRACTED · 45% INFERRED · 0% AMBIGUOUS · INFERRED: 2333 edges (avg confidence: 0.7)
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

## God Nodes (most connected - your core abstractions)
1. `NumpyRidgeBaseline` - 75 edges
2. `Data schemas, split manifests, and leakage guards.` - 59 edges
3. `NeuralStateSpaceTranslator` - 50 edges
4. `DualFieldConfig` - 50 edges
5. `TorchMLPBaseline` - 48 edges
6. `NeuroTwinPairOperator` - 48 edges
7. `TinyTransformerBaseline` - 48 edges
8. `TinySSMBaseline` - 48 edges
9. `NeuralStateSpaceTranslatorConfig` - 44 edges
10. `build_split_manifest()` - 40 edges

## Surprising Connections (you probably didn't know these)
- `Operator-recovery falsification diagnostics for the v3 Transition Gym.  PROPOSED` --uses--> `Outcome`  [INFERRED]
  /Users/aayu/conductor/workspaces/kahlus-v2/irvine/src/neurotwin/transition_gym/operator_recovery.py → /Users/aayu/conductor/workspaces/kahlus-v2/irvine/src/neurotwin/falsification.py
- `Latent pre-perturbation state z_pre per episode (E, Dz).` --uses--> `Outcome`  [INFERRED]
  /Users/aayu/conductor/workspaces/kahlus-v2/irvine/src/neurotwin/transition_gym/operator_recovery.py → /Users/aayu/conductor/workspaces/kahlus-v2/irvine/src/neurotwin/falsification.py
- `Latent pre-perturbation state z_pre per episode (E, Dz).` --uses--> `Outcome`  [INFERRED]
  /Users/aayu/conductor/workspaces/kahlus-v2/irvine/src/neurotwin/transition_gym/operator_recovery.py → /Users/aayu/conductor/workspaces/kahlus-v2/irvine/src/neurotwin/falsification.py
- `Recover each hidden operator ``M_k`` from latent transitions and compare to trut` --uses--> `Outcome`  [INFERRED]
  /Users/aayu/conductor/workspaces/kahlus-v2/irvine/src/neurotwin/transition_gym/operator_recovery.py → /Users/aayu/conductor/workspaces/kahlus-v2/irvine/src/neurotwin/falsification.py
- `Predict held-out AB/BA compositions from single-operator estimates only.      Si` --uses--> `Outcome`  [INFERRED]
  /Users/aayu/conductor/workspaces/kahlus-v2/irvine/src/neurotwin/transition_gym/operator_recovery.py → /Users/aayu/conductor/workspaces/kahlus-v2/irvine/src/neurotwin/falsification.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.03
Nodes (131): ArchitectureSpec, BaselineRunResult, _fit_ridge(), _flatten_window(), _last_step(), _predict_torch(), Shared local baseline runner for the v2 dual-field and v3 Transition Gym tasks., Dependency-free retrieval baseline: mean target of the k nearest train rows by L (+123 more)

### Community 1 - "Community 1"
Cohesion: 0.02
Nodes (96): transition_gym_regression_task(), benchmark_report(), v2 dual-field synthetic falsification benchmark.  Runs the diagnostic battery, t, Score the untrained KTM scaffold on the same response-profile task (informationa, run_v3_benchmark(), _score_ktm(), V3BenchmarkResult, write_v2_report() (+88 more)

### Community 2 - "Community 2"
Cohesion: 0.03
Nodes (108): AlgonautsPrepareResult, Prepare Algonauts 2025 stimulus-to-fMRI batches from verified local artifacts., audit_prepared_eval_inputs(), audit_split_manifest(), AuditReport, _duplicate_metadata_value_violations(), _event_metadata_violations(), _forbidden_event_metadata_violations() (+100 more)

### Community 3 - "Community 3"
Cohesion: 0.03
Nodes (148): ensure_src_import_path(), Allow direct script execution without mutating imports at module load., main(), _nccl_version(), _payload(), _positive_int(), evaluate_debug_gate(), _finite_number() (+140 more)

### Community 4 - "Community 4"
Cohesion: 0.02
Nodes (56): BaseObservationOperator, Base class for latent-field-to-observation operators., BaseObservationOperator, BehaviorObservationOperator, Compile a latent neural field into behavior or task-label predictions., _expert_utilization(), NeuralFieldCompiler, NeuralFieldCompilerConfig (+48 more)

### Community 5 - "Community 5"
Cohesion: 0.04
Nodes (83): dual_field_regression_task(), run_baselines(), run_v2_benchmark(), V2BenchmarkResult, DualFieldConfig, Configuration for the Kahlus v2 synthetic dual-field scaffold.  PROPOSED / SYNTH, Shapes and dynamics constants for the dual-field synthetic system.      The syst, hrf_lag_weights() (+75 more)

### Community 6 - "Community 6"
Cohesion: 0.04
Nodes (59): _cmd_train(), _config_value(), _csv_cell(), _csv_rows(), _dry_run_result(), _finite_number(), _has_prepared_training_inputs(), _pair_operator_ablation_csv() (+51 more)

### Community 7 - "Community 7"
Cohesion: 0.05
Nodes (68): _mapping(), _optional_int(), _optional_nonnegative_float(), PreparedDataConfig, PreparedModelConfig, PreparedTrainingConfigInput, PreparedTrainingSectionConfig, _resolve_modalities() (+60 more)

### Community 8 - "Community 8"
Cohesion: 0.05
Nodes (40): _add_eval_audit_args(), _add_eval_demo_args(), _add_eval_manifest_args(), _add_eval_suite_args(), _add_eval_window_args(), _cmd_cluster_materialize_config(), _cmd_cluster_preflight(), _cmd_doctor() (+32 more)

### Community 9 - "Community 9"
Cohesion: 0.04
Nodes (43): format_artifact_report_md(), Stage 0 artifact audit: does environment/device change affect EEG hardware (no b, Synthesize a phantom/idle EEG-like recording ``(n_channels, n_samples)``.      B, Compute artifact features for both conditions and summarize the descriptive resp, run_artifact_audit(), synthesize_idle_recording(), band_power(), channel_artifact_features() (+35 more)

### Community 10 - "Community 10"
Cohesion: 0.07
Nodes (51): _aggregate_rank_from_payload(), _aggregate_rank_matches(), aggregate_seed_metrics(), aggregate_seed_ranks(), aggregated_seed_tasks(), AggregateRankRecord, _audit_payload(), build_paper_mode_evidence() (+43 more)

### Community 11 - "Community 11"
Cohesion: 0.08
Nodes (31): ensure_scripts_import_path(), Allow lazy sibling script imports from module-based test loaders., _bundle_rel_parts(), copy_bundle_file(), copy_current_docker_log(), copy_current_run_logs(), copy_source_file(), copy_tree_files() (+23 more)

### Community 12 - "Community 12"
Cohesion: 0.07
Nodes (34): regression_metrics(), _metrics(), bandpower_error(), bootstrap_ci(), mae(), mse(), pearsonr(), r2_score() (+26 more)

### Community 13 - "Community 13"
Cohesion: 0.07
Nodes (39): _aggregate_seed_payloads(), _all_metrics_are_finite(), _criterion(), _evidence_gate(), _falsification(), _fit_ridge(), _fit_sequence_baseline(), _format_diagnostic_report() (+31 more)

### Community 14 - "Community 14"
Cohesion: 0.1
Nodes (35): _align_response_and_stimulus(), _cached_hash(), _candidate_feature_files(), _candidate_response_files(), _canonical_stimulus_id(), _compatible_time_length(), _discover_response_records(), _embedded_stimulus_features() (+27 more)

### Community 15 - "Community 15"
Cohesion: 0.07
Nodes (31): append_artifact_errors(), baseline_ranking_rows(), csv_cell(), csv_rows(), diagnostic_status(), first_json_artifact(), format_aggregate_rank(), is_artifact_error() (+23 more)

### Community 16 - "Community 16"
Cohesion: 0.08
Nodes (17): architecture_registry(), architecture_spec(), architecture_status(), build_architecture_model(), estimate_architecture_extra_parameters(), _nfc_factory(), normalize_architecture_type(), _normalize_key() (+9 more)

### Community 17 - "Community 17"
Cohesion: 0.11
Nodes (36): _dataset_site_generalization_from_windows(), _format_stimulus_evidence(), _group_windows(), run_prepared_auxiliary_tasks(), _scope_status(), _stimulus_evidence_from_tasks(), _subject_adaptation_from_windows(), _task_result_to_dict() (+28 more)

### Community 18 - "Community 18"
Cohesion: 0.12
Nodes (25): format_prepared_eval_audit(), _cmd_eval(), EvalCommandConfig, EvalCommandResult, _manifest_paths(), _paper_demo_config(), _paper_demo_error(), _paper_demo_exit_code() (+17 more)

### Community 19 - "Community 19"
Cohesion: 0.14
Nodes (2): ExpandedCliTests, _valid_paper_mode_gate()

### Community 20 - "Community 20"
Cohesion: 0.17
Nodes (18): bids_manifest_summary(), _events_for(), _infer_modality(), _is_bids_signal(), _listlike(), _load_timeseries_derivative(), _parse_entities(), _read_tsv() (+10 more)

### Community 21 - "Community 21"
Cohesion: 0.17
Nodes (12): assemble_gate(), build_report(), _iter_numbers(), outcome_dicts(), outcomes_finite(), Shared falsification-benchmark harness for the synthetic v-lane benchmarks.  A n, Assemble the standard falsification report; ``extra`` adds lane-specific keys., Yield every scalar number nested anywhere inside a detail value. (+4 more)

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
Cohesion: 0.32
Nodes (7): commutator(), commutator_matrix(), commutator_norm(), Commutators of the latent perturbation operators: [Ta, Tb] = Ta Tb - Tb Ta.  A n, Matrix commutator ``A B - B A``., Frobenius norm of the commutator (0 iff the operators commute)., Pairwise commutator-norm matrix for a set of operators (``(K, K)``).

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
Nodes (1): Least-squares affine map ``y ≈ x @ W.T + c``; returns (W, c).

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): Recover each hidden operator ``M_k`` from latent transitions and compare to trut

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): Predict held-out AB/BA compositions from single-operator estimates only.      Si

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Explicit AB-vs-BA gap; the battery must be genuinely non-commutative.

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): Mean trajectory, operator-induced, and subject-transfer response-profile distanc

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): Evaluate the gate and return the dossier-schema payload.      All checks are con

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Persist a gate payload as pretty, sorted JSON (reuses repro.write_json).

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Load a gate payload written by :func:`write_evidence_gate`.

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Narrow loaded YAML at the command boundary after load_config validation.

## Knowledge Gaps
- **116 isolated node(s):** `Allow direct script execution without mutating imports at module load.`, `Allow lazy sibling script imports from module-based test loaders.`, `Narrow loaded YAML at the command boundary after load_config validation.`, `Raised when an experiment config cannot be loaded or validated.`, `Resolve a source commit from git, falling back to COMMIT_HASH.txt.` (+111 more)
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
- **Thin community `Community 43`** (1 nodes): `Least-squares affine map ``y ≈ x @ W.T + c``; returns (W, c).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `Recover each hidden operator ``M_k`` from latent transitions and compare to trut`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Predict held-out AB/BA compositions from single-operator estimates only.      Si`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Explicit AB-vs-BA gap; the battery must be genuinely non-commutative.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `Mean trajectory, operator-induced, and subject-transfer response-profile distanc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `Evaluate the gate and return the dossier-schema payload.      All checks are con`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Persist a gate payload as pretty, sorted JSON (reuses repro.write_json).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Load a gate payload written by :func:`write_evidence_gate`.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Narrow loaded YAML at the command boundary after load_config validation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Data schemas, split manifests, and leakage guards.` connect `Community 4` to `Community 0`, `Community 1`, `Community 5`, `Community 9`, `Community 12`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **Why does `PerturbationLibrary` connect `Community 1` to `Community 3`, `Community 12`, `Community 4`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **Why does `build_split_manifest()` connect `Community 2` to `Community 4`, `Community 6`, `Community 8`, `Community 12`, `Community 14`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Are the 114 inferred relationships involving `ValueError` (e.g. with `_require_bundle_rel()` and `resolve_prepared_config()`) actually correct?**
  _`ValueError` has 114 INFERRED edges - model-reasoned connections that need verification._
- **Are the 70 inferred relationships involving `NumpyRidgeBaseline` (e.g. with `DualFieldTests` and `BaselinesAndArchitectureTests`) actually correct?**
  _`NumpyRidgeBaseline` has 70 INFERRED edges - model-reasoned connections that need verification._
- **Are the 41 inferred relationships involving `Data schemas, split manifests, and leakage guards.` (e.g. with `EMContext` and `IdleRecordingMetadata`) actually correct?**
  _`Data schemas, split manifests, and leakage guards.` has 41 INFERRED edges - model-reasoned connections that need verification._
- **Are the 43 inferred relationships involving `NeuralStateSpaceTranslator` (e.g. with `ModelMetadataGeometryTests` and `ModelShapeTests`) actually correct?**
  _`NeuralStateSpaceTranslator` has 43 INFERRED edges - model-reasoned connections that need verification._