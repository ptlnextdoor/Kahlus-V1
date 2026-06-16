# Graph Report - /Users/aayu/conductor/workspaces/kahlus-v2/surat  (2026-06-16)

## Corpus Check
- 243 files · ~311,241 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2391 nodes · 6218 edges · 57 communities detected
- Extraction: 54% EXTRACTED · 46% INFERRED · 0% AMBIGUOUS · INFERRED: 2832 edges (avg confidence: 0.7)
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

## God Nodes (most connected - your core abstractions)
1. `NumpyRidgeBaseline` - 89 edges
2. `Data schemas, split manifests, and leakage guards.` - 69 edges
3. `TorchMLPBaseline` - 63 edges
4. `TinyTransformerBaseline` - 63 edges
5. `TinySSMBaseline` - 63 edges
6. `DistributedInfo` - 51 edges
7. `NeuralStateSpaceTranslator` - 50 edges
8. `DualFieldConfig` - 50 edges
9. `NeuroTwinPairOperator` - 48 edges
10. `train_ktm()` - 48 edges

## Surprising Connections (you probably didn't know these)
- `Kahlus-EM Stage 0 artifact report generator (no-human, synthetic).  Turns a Stag` --uses--> `Outcome`  [INFERRED]
  /Users/aayu/conductor/workspaces/kahlus-v2/surat/src/neurotwin/em/stage0_report.py → /Users/aayu/conductor/workspaces/kahlus-v2/surat/src/neurotwin/falsification.py
- `Assemble the Stage 0 report bundle (severity, contamination map, gate, report di` --uses--> `Outcome`  [INFERRED]
  /Users/aayu/conductor/workspaces/kahlus-v2/surat/src/neurotwin/em/stage0_report.py → /Users/aayu/conductor/workspaces/kahlus-v2/surat/src/neurotwin/falsification.py
- `A plain pass/fail recommendation combining gate validity and severity verdict.` --uses--> `Outcome`  [INFERRED]
  /Users/aayu/conductor/workspaces/kahlus-v2/surat/src/neurotwin/em/stage0_report.py → /Users/aayu/conductor/workspaces/kahlus-v2/surat/src/neurotwin/falsification.py
- `Render the Stage 0 artifact report as Markdown with all required sections.` --uses--> `Outcome`  [INFERRED]
  /Users/aayu/conductor/workspaces/kahlus-v2/surat/src/neurotwin/em/stage0_report.py → /Users/aayu/conductor/workspaces/kahlus-v2/surat/src/neurotwin/falsification.py
- `Write the Markdown report + shared-core report/gate JSONs; return their paths.` --uses--> `Outcome`  [INFERRED]
  /Users/aayu/conductor/workspaces/kahlus-v2/surat/src/neurotwin/em/stage0_report.py → /Users/aayu/conductor/workspaces/kahlus-v2/surat/src/neurotwin/falsification.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.02
Nodes (210): append_artifact_errors(), baseline_ranking_rows(), csv_cell(), csv_rows(), diagnostic_status(), first_json_artifact(), format_aggregate_rank(), is_artifact_error() (+202 more)

### Community 1 - "Community 1"
Cohesion: 0.03
Nodes (151): ArchitectureSpec, _build_torch_baseline(), _clone_state(), _f32(), _fit_ridge(), _flatten_window(), _last_step(), _predict_state() (+143 more)

### Community 2 - "Community 2"
Cohesion: 0.02
Nodes (138): baseline_table_rows(), BaselineRunResult, _all_finite(), _assess_recovery(), _checkpoint_files(), Output-bundle writer for the v3 KTM training harness (PROPOSED / SYNTHETIC ONLY), Recovery-scope blockers + red-team dossier + selection-parity record (pure; no I, write_training_bundle() (+130 more)

### Community 3 - "Community 3"
Cohesion: 0.03
Nodes (118): audit_prepared_eval_inputs(), audit_split_manifest(), AuditReport, _duplicate_metadata_value_violations(), _event_metadata_violations(), _forbidden_event_metadata_violations(), _forbidden_metadata(), format_prepared_eval_audit() (+110 more)

### Community 4 - "Community 4"
Cohesion: 0.02
Nodes (100): transition_gym_regression_task(), benchmark_report(), Score the untrained KTM scaffold on the same response-profile task (informationa, run_v3_benchmark(), _score_ktm(), V3BenchmarkResult, KTMConfig, build_data_card() (+92 more)

### Community 5 - "Community 5"
Cohesion: 0.02
Nodes (99): format_artifact_report_md(), Stage 0 artifact audit: does environment/device change affect EEG hardware (no b, Synthesize a phantom/idle EEG-like recording ``(n_channels, n_samples)``.      B, Compute artifact features for both conditions and summarize the descriptive resp, run_artifact_audit(), synthesize_idle_recording(), artifact_severity_summary(), band_contamination_score() (+91 more)

### Community 6 - "Community 6"
Cohesion: 0.03
Nodes (84): dual_field_regression_task(), run_baselines(), v2 dual-field synthetic falsification benchmark.  Runs the diagnostic battery, t, run_v2_benchmark(), V2BenchmarkResult, DualFieldConfig, Shapes and dynamics constants for the dual-field synthetic system.      The syst, hrf_lag_weights() (+76 more)

### Community 7 - "Community 7"
Cohesion: 0.03
Nodes (80): architecture_registry(), architecture_spec(), architecture_status(), build_architecture_model(), estimate_architecture_extra_parameters(), normalize_architecture_type(), _normalize_key(), main() (+72 more)

### Community 8 - "Community 8"
Cohesion: 0.03
Nodes (63): _nfc_factory(), _pair_operator_factory(), _translator_factory(), _cmd_train(), _config_value(), _csv_cell(), _csv_rows(), _dry_run_result() (+55 more)

### Community 9 - "Community 9"
Cohesion: 0.04
Nodes (84): AlgonautsPrepareResult, _align_response_and_stimulus(), _cached_hash(), _candidate_feature_files(), _candidate_response_files(), _canonical_stimulus_id(), _compatible_time_length(), _discover_response_records() (+76 more)

### Community 10 - "Community 10"
Cohesion: 0.03
Nodes (36): BaseObservationOperator, Base class for latent-field-to-observation operators., BaseObservationOperator, BehaviorObservationOperator, Compile a latent neural field into behavior or task-label predictions., _expert_utilization(), NeuralFieldCompiler, NeuralFieldCompilerConfig (+28 more)

### Community 11 - "Community 11"
Cohesion: 0.04
Nodes (49): _add_eval_audit_args(), _add_eval_demo_args(), _add_eval_manifest_args(), _add_eval_suite_args(), _add_eval_window_args(), _cmd_cluster_materialize_config(), _cmd_cluster_preflight(), _cmd_data_audit() (+41 more)

### Community 12 - "Community 12"
Cohesion: 0.07
Nodes (34): regression_metrics(), _metrics(), bandpower_error(), bootstrap_ci(), mae(), mse(), pearsonr(), r2_score() (+26 more)

### Community 13 - "Community 13"
Cohesion: 0.08
Nodes (31): ensure_scripts_import_path(), Allow lazy sibling script imports from module-based test loaders., _bundle_rel_parts(), copy_bundle_file(), copy_current_docker_log(), copy_current_run_logs(), copy_source_file(), copy_tree_files() (+23 more)

### Community 14 - "Community 14"
Cohesion: 0.09
Nodes (37): audit_claims(), audit_environment(), audit_evidence(), audit_metrics(), _audit_root(), AuditResult, check_required_files(), _find_bundle_root() (+29 more)

### Community 15 - "Community 15"
Cohesion: 0.07
Nodes (40): _aggregate_seed_payloads(), _all_metrics_are_finite(), _criterion(), _evidence_gate(), _falsification(), _fit_ridge(), _fit_sequence_baseline(), _format_diagnostic_report() (+32 more)

### Community 16 - "Community 16"
Cohesion: 0.12
Nodes (14): _cmd_report(), _cmd_run(), copy_paper_mode_artifacts(), copy_prepared_eval_audit(), finalize_run(), paper_mode_gate_passed(), run_paper_diagnostics(), RunFinalizeConfig (+6 more)

### Community 17 - "Community 17"
Cohesion: 0.15
Nodes (2): ExpandedCliTests, _valid_paper_mode_gate()

### Community 18 - "Community 18"
Cohesion: 0.15
Nodes (14): _build_events(), from_checkpoint(), from_pretrained(), _load_local_config(), _normal_stimulus_modality(), Build minimal local event rows for smoke/pipeline tests.          Text events ar, Return local event-row dictionaries for smoke/pipeline tests.          Prefer th, Compatibility shim returning local event rows, not a pandas DataFrame. (+6 more)

### Community 19 - "Community 19"
Cohesion: 0.19
Nodes (22): build_future_forecasting_task_from_windows(), _cross_modal_task_from_windows(), _future_task_from_windows(), _future_xy(), _local_stimulus_artifact_path(), _metadata_list(), _normalize_hash(), _optional_text() (+14 more)

### Community 20 - "Community 20"
Cohesion: 0.22
Nodes (8): Adversarial falsification of a KTM ``synthetic_ktm_recovery`` candidate (PROPOSE, Decide whether a recovery candidate survives the red-team battery.      Returns, Mean/std/min/max + a conservative normal-approx lower bound on relative improvem, recovery_redteam_gate(), seed_summary(), _families(), RecoveryRedteamGateTests, _seeds()

### Community 21 - "Community 21"
Cohesion: 0.13
Nodes (1): ArtifactDocsContractsTests

### Community 22 - "Community 22"
Cohesion: 0.19
Nodes (7): dataset_registry(), DatasetAdapterSpec, RegistryTests, permissive_upstreams(), quarantined_upstreams(), upstream_registry(), UpstreamSpec

### Community 23 - "Community 23"
Cohesion: 0.23
Nodes (4): assert_runner_archive(), copy_repo_to_temp_git(), HandoffZipArtifactTests, RunnerBundleArtifactTests

### Community 24 - "Community 24"
Cohesion: 0.25
Nodes (3): baseline_catalog_rows(), BaselineCatalogEntry, _baseline_catalog()

### Community 25 - "Community 25"
Cohesion: 0.46
Nodes (7): evaluate_debug_gate(), _finite_number(), _float_or_nan(), main(), _nonfinite_number_paths(), _read_json(), _task_result()

### Community 26 - "Community 26"
Cohesion: 0.4
Nodes (1): KtmHandoffTests

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
Nodes (0):

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): Load a local NeuroTwin TRIBE-style config or seeded defaults.          No pretra

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): Compatibility shim for TRIBE-style callers.          Prefer :meth:`from_checkpoi

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): Evaluate the gate and return the dossier-schema payload.      All checks are con

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): Persist a gate payload as pretty, sorted JSON (reuses repro.write_json).

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Roll the autonomous base dynamics to produce per-episode history states.      Re

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): Roll the base dynamics forward ``horizon`` steps from a perturbed state.      Re

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): Least-squares affine map ``y ≈ x @ W.T + c``; returns (W, c).

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Recover each hidden operator ``M_k`` from latent transitions and compare to trut

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Predict held-out AB/BA compositions from single-operator estimates only.      Si

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Explicit AB-vs-BA gap; the battery must be genuinely non-commutative.

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): Mean trajectory, operator-induced, and subject-transfer response-profile distanc

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): Evaluate the gate and return the dossier-schema payload.      All checks are con

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): Persist a gate payload as pretty, sorted JSON (reuses repro.write_json).

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): Load a gate payload written by :func:`write_evidence_gate`.

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): Narrow loaded YAML at the command boundary after load_config validation.

## Knowledge Gaps
- **156 isolated node(s):** `Tests for the KTM A100 evidence intake auditor (synthetic fixtures only).`, `Build a synthetic KTM A100 evidence bundle folder; returns the bundle root.`, `Base (no-ablation) failure-analysis report shape, finiteness, and gate disciplin`, `The ablation matrix loads and smoke-runs, and never earns recovery.`, `Each shipped ablation / Sprint 3C YAML loads into a valid KTMTrainConfig.` (+151 more)
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
- **Thin community `Community 40`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `paper_mode.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Load a local NeuroTwin TRIBE-style config or seeded defaults.          No pretra`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Compatibility shim for TRIBE-style callers.          Prefer :meth:`from_checkpoi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `Evaluate the gate and return the dossier-schema payload.      All checks are con`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Persist a gate payload as pretty, sorted JSON (reuses repro.write_json).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Roll the autonomous base dynamics to produce per-episode history states.      Re`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `Roll the base dynamics forward ``horizon`` steps from a perturbed state.      Re`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `Least-squares affine map ``y ≈ x @ W.T + c``; returns (W, c).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Recover each hidden operator ``M_k`` from latent transitions and compare to trut`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Predict held-out AB/BA compositions from single-operator estimates only.      Si`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Explicit AB-vs-BA gap; the battery must be genuinely non-commutative.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Mean trajectory, operator-induced, and subject-transfer response-profile distanc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Evaluate the gate and return the dossier-schema payload.      All checks are con`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `Persist a gate payload as pretty, sorted JSON (reuses repro.write_json).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `Load a gate payload written by :func:`write_evidence_gate`.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `Narrow loaded YAML at the command boundary after load_config validation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Data schemas, split manifests, and leakage guards.` connect `Community 10` to `Community 1`, `Community 2`, `Community 4`, `Community 5`, `Community 6`, `Community 12`, `Community 14`, `Community 18`?**
  _High betweenness centrality (0.102) - this node is a cross-community bridge._
- **Why does `train_ktm()` connect `Community 2` to `Community 0`, `Community 1`, `Community 3`, `Community 4`, `Community 5`, `Community 7`, `Community 8`, `Community 9`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Why does `KTM` connect `Community 4` to `Community 10`, `Community 6`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Are the 122 inferred relationships involving `ValueError` (e.g. with `_parse_seeds()` and `_require_bundle_rel()`) actually correct?**
  _`ValueError` has 122 INFERRED edges - model-reasoned connections that need verification._
- **Are the 84 inferred relationships involving `NumpyRidgeBaseline` (e.g. with `DualFieldTests` and `BaselinesAndArchitectureTests`) actually correct?**
  _`NumpyRidgeBaseline` has 84 INFERRED edges - model-reasoned connections that need verification._
- **Are the 49 inferred relationships involving `Data schemas, split manifests, and leakage guards.` (e.g. with `EMContext` and `IdleRecordingMetadata`) actually correct?**
  _`Data schemas, split manifests, and leakage guards.` has 49 INFERRED edges - model-reasoned connections that need verification._
- **Are the 59 inferred relationships involving `TorchMLPBaseline` (e.g. with `BaselinesAndArchitectureTests` and `RegressionTask`) actually correct?**
  _`TorchMLPBaseline` has 59 INFERRED edges - model-reasoned connections that need verification._