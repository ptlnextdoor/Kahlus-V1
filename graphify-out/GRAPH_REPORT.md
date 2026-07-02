# Graph Report - kahlus_trial0_m0_clean_worktree  (2026-07-02)

## Corpus Check
- 263 files · ~172,091 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2690 nodes · 7033 edges · 48 communities detected
- Extraction: 57% EXTRACTED · 43% INFERRED · 0% AMBIGUOUS · INFERRED: 3051 edges (avg confidence: 0.7)
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

## God Nodes (most connected - your core abstractions)
1. `NumpyRidgeBaseline` - 96 edges
2. `Data schemas, split manifests, and leakage guards.` - 74 edges
3. `TorchMLPBaseline` - 63 edges
4. `TinyTransformerBaseline` - 63 edges
5. `TinySSMBaseline` - 63 edges
6. `DistributedInfo` - 51 edges
7. `NeuralStateSpaceTranslator` - 50 edges
8. `DualFieldConfig` - 50 edges
9. `NeuroTwinPairOperator` - 48 edges
10. `train_ktm()` - 48 edges

## Surprising Connections (you probably didn't know these)
- `Outcome` --uses--> `Kahlus-EM Stage 0 artifact report generator (no-human, synthetic).  Turns a Stag`  [INFERRED]
  src/neurotwin/falsification.py → /Users/aayu/conductor/workspaces/kahlus-v2/surat/src/neurotwin/em/stage0_report.py
- `Outcome` --uses--> `Assemble the Stage 0 report bundle (severity, contamination map, gate, report di`  [INFERRED]
  src/neurotwin/falsification.py → /Users/aayu/conductor/workspaces/kahlus-v2/surat/src/neurotwin/em/stage0_report.py
- `Outcome` --uses--> `A plain pass/fail recommendation combining gate validity and severity verdict.`  [INFERRED]
  src/neurotwin/falsification.py → /Users/aayu/conductor/workspaces/kahlus-v2/surat/src/neurotwin/em/stage0_report.py
- `Outcome` --uses--> `Render the Stage 0 artifact report as Markdown with all required sections.`  [INFERRED]
  src/neurotwin/falsification.py → /Users/aayu/conductor/workspaces/kahlus-v2/surat/src/neurotwin/em/stage0_report.py
- `Outcome` --uses--> `Write the Markdown report + shared-core report/gate JSONs; return their paths.`  [INFERRED]
  src/neurotwin/falsification.py → /Users/aayu/conductor/workspaces/kahlus-v2/surat/src/neurotwin/em/stage0_report.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.02
Nodes (251): append_artifact_errors(), baseline_ranking_rows(), csv_cell(), csv_rows(), diagnostic_status(), first_json_artifact(), format_aggregate_rank(), is_artifact_error() (+243 more)

### Community 1 - "Community 1"
Cohesion: 0.02
Nodes (157): build_fewshot_adaptation_task(), FewShotAdaptationTask, SubjectAdaptationSplit, AlgonautsPrepareResult, _align_response_and_stimulus(), _cached_hash(), _candidate_feature_files(), _candidate_response_files() (+149 more)

### Community 2 - "Community 2"
Cohesion: 0.02
Nodes (142): baseline_table_rows(), _all_finite(), _assess_recovery(), _checkpoint_files(), Output-bundle writer for the v3 KTM training harness (PROPOSED / SYNTHETIC ONLY), Recovery-scope blockers + red-team dossier + selection-parity record (pure; no I, write_training_bundle(), apply_resume() (+134 more)

### Community 3 - "Community 3"
Cohesion: 0.03
Nodes (157): ArchitectureSpec, BaselineRunResult, _build_torch_baseline(), _clone_state(), _f32(), _fit_ridge(), _flatten_window(), _last_step() (+149 more)

### Community 4 - "Community 4"
Cohesion: 0.02
Nodes (95): transition_gym_regression_task(), benchmark_report(), Score the untrained KTM scaffold on the same response-profile task (informationa, run_v3_benchmark(), _score_ktm(), V3BenchmarkResult, KTMConfig, build_data_card() (+87 more)

### Community 5 - "Community 5"
Cohesion: 0.03
Nodes (89): architecture_registry(), architecture_spec(), architecture_status(), build_architecture_model(), estimate_architecture_extra_parameters(), _nfc_factory(), normalize_architecture_type(), _normalize_key() (+81 more)

### Community 6 - "Community 6"
Cohesion: 0.03
Nodes (84): dual_field_regression_task(), run_baselines(), v2 dual-field synthetic falsification benchmark.  Runs the diagnostic battery, t, run_v2_benchmark(), V2BenchmarkResult, DualFieldConfig, Shapes and dynamics constants for the dual-field synthetic system.      The syst, hrf_lag_weights() (+76 more)

### Community 7 - "Community 7"
Cohesion: 0.04
Nodes (86): A single room/device/environment passive-logging entry., RoomEnvironmentLog, _cluster_bootstrap_rfs(), _crossfit_proba(), _crossfit_residual_proba(), discrete_survival_labels(), _fit_predict(), _fit_residual_offset_predict() (+78 more)

### Community 8 - "Community 8"
Cohesion: 0.03
Nodes (37): BaseObservationOperator, Base class for latent-field-to-observation operators., BaseObservationOperator, BehaviorObservationOperator, Compile a latent neural field into behavior or task-label predictions., _expert_utilization(), NeuralFieldCompiler, _observation_operator() (+29 more)

### Community 9 - "Community 9"
Cohesion: 0.04
Nodes (64): write_baseline_table(), write_run_artifacts(), write_v2_report(), write_v3_report(), _cmd_estimate(), _config_value(), _csv_cell(), _csv_rows() (+56 more)

### Community 10 - "Community 10"
Cohesion: 0.04
Nodes (59): format_artifact_report_md(), Stage 0 artifact audit: does environment/device change affect EEG hardware (no b, Synthesize a phantom/idle EEG-like recording ``(n_channels, n_samples)``.      B, Compute artifact features for both conditions and summarize the descriptive resp, run_artifact_audit(), synthesize_idle_recording(), artifact_severity_summary(), band_contamination_score() (+51 more)

### Community 11 - "Community 11"
Cohesion: 0.06
Nodes (78): _aggregate_classification_inflation(), _aggregate_classification_interpretation(), _aggregate_classification_seed_results(), _aggregate_identity_risk(), _aggregate_identity_seed_results(), _aggregate_leakage_interpretation(), _aggregate_leakage_seed_results(), _bad_segment_split_classification_metrics() (+70 more)

### Community 12 - "Community 12"
Cohesion: 0.05
Nodes (42): _add_eval_audit_args(), _add_eval_demo_args(), _add_eval_manifest_args(), _add_eval_suite_args(), _add_eval_window_args(), _cmd_cluster_materialize_config(), _cmd_cluster_preflight(), _cmd_data_audit() (+34 more)

### Community 13 - "Community 13"
Cohesion: 0.07
Nodes (35): _metric_bundle(), regression_metrics(), _metrics(), bandpower_error(), bootstrap_ci(), mae(), mse(), pearsonr() (+27 more)

### Community 14 - "Community 14"
Cohesion: 0.08
Nodes (31): ensure_scripts_import_path(), Allow lazy sibling script imports from module-based test loaders., _bundle_rel_parts(), copy_bundle_file(), copy_current_docker_log(), copy_current_run_logs(), copy_source_file(), copy_tree_files() (+23 more)

### Community 15 - "Community 15"
Cohesion: 0.08
Nodes (38): main(), audit_claims(), audit_environment(), audit_evidence(), audit_metrics(), _audit_root(), AuditResult, check_required_files() (+30 more)

### Community 16 - "Community 16"
Cohesion: 0.07
Nodes (39): _aggregate_seed_payloads(), _all_metrics_are_finite(), _criterion(), _evidence_gate(), _falsification(), _fit_ridge(), _fit_sequence_baseline(), _format_diagnostic_report() (+31 more)

### Community 17 - "Community 17"
Cohesion: 0.08
Nodes (23): assemble_gate(), build_report(), _iter_numbers(), outcome_dicts(), outcomes_finite(), Shared falsification-benchmark harness for the synthetic v-lane benchmarks.  A n, Assemble the standard falsification report; ``extra`` adds lane-specific keys., Yield every scalar number nested anywhere inside a detail value. (+15 more)

### Community 18 - "Community 18"
Cohesion: 0.11
Nodes (36): _dataset_site_generalization_from_windows(), _format_stimulus_evidence(), _group_windows(), run_prepared_auxiliary_tasks(), _scope_status(), _stimulus_evidence_from_tasks(), _subject_adaptation_from_windows(), _task_result_to_dict() (+28 more)

### Community 19 - "Community 19"
Cohesion: 0.09
Nodes (24): _cmd_report(), _cmd_run(), _baseline_ranking_present(), build_prepared_evidence_gate(), _competitor_reproduction_status_present(), format_evidence_diagnostic_report(), _paper_mode_gate_present(), _prepared_suite_has_rankings() (+16 more)

### Community 20 - "Community 20"
Cohesion: 0.11
Nodes (30): _adaptation_artifact_index_rows(), _adaptation_baseline_gap_summary(), _adaptation_dataset_summary(), _adaptation_subject_baseline_gap_summary(), _adaptation_target_scale_context(), _adaptation_verification_payload(), _AdapterSequenceForecaster, audit_adaptation_checksum_manifest() (+22 more)

### Community 21 - "Community 21"
Cohesion: 0.12
Nodes (25): format_prepared_eval_audit(), _cmd_eval(), EvalCommandConfig, EvalCommandResult, _manifest_paths(), _paper_demo_config(), _paper_demo_error(), _paper_demo_exit_code() (+17 more)

### Community 22 - "Community 22"
Cohesion: 0.15
Nodes (19): bids_manifest_summary(), _events_for(), _infer_modality(), _is_bids_signal(), _listlike(), _load_timeseries_derivative(), _parse_entities(), _read_tsv() (+11 more)

### Community 23 - "Community 23"
Cohesion: 0.15
Nodes (2): ExpandedCliTests, _valid_paper_mode_gate()

### Community 24 - "Community 24"
Cohesion: 0.15
Nodes (14): _build_events(), from_checkpoint(), from_pretrained(), _load_local_config(), _normal_stimulus_modality(), Build minimal local event rows for smoke/pipeline tests.          Text events ar, Return local event-row dictionaries for smoke/pipeline tests.          Prefer th, Compatibility shim returning local event rows, not a pandas DataFrame. (+6 more)

### Community 25 - "Community 25"
Cohesion: 0.17
Nodes (18): _as_external_dataset_record(), _environment_payload(), _fmt(), _freeze_manifests(), _git(), _git_state(), _manifest_payload(), _read_csv_rows() (+10 more)

### Community 26 - "Community 26"
Cohesion: 0.13
Nodes (1): ArtifactDocsContractsTests

### Community 27 - "Community 27"
Cohesion: 0.19
Nodes (7): dataset_registry(), DatasetAdapterSpec, RegistryTests, permissive_upstreams(), quarantined_upstreams(), upstream_registry(), UpstreamSpec

### Community 28 - "Community 28"
Cohesion: 0.23
Nodes (4): assert_runner_archive(), copy_repo_to_temp_git(), HandoffZipArtifactTests, RunnerBundleArtifactTests

### Community 29 - "Community 29"
Cohesion: 0.25
Nodes (3): baseline_catalog_rows(), BaselineCatalogEntry, _baseline_catalog()

### Community 30 - "Community 30"
Cohesion: 0.32
Nodes (7): commutator(), commutator_matrix(), commutator_norm(), Commutators of the latent perturbation operators: [Ta, Tb] = Ta Tb - Tb Ta.  A n, Matrix commutator ``A B - B A``., Frobenius norm of the commutator (0 iff the operators commute)., Pairwise commutator-norm matrix for a set of operators (``(K, K)``).

### Community 31 - "Community 31"
Cohesion: 0.4
Nodes (1): KtmHandoffTests

### Community 32 - "Community 32"
Cohesion: 0.67
Nodes (2): Second-difference smoothness penalty for future EEG windows., smoothness_loss()

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): Load a local NeuroTwin TRIBE-style config or seeded defaults.          No pretra

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Compatibility shim for TRIBE-style callers.          Prefer :meth:`from_checkpoi

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Evaluate the gate and return the dossier-schema payload.      All checks are con

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Persist a gate payload as pretty, sorted JSON (reuses repro.write_json).

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): Roll the autonomous base dynamics to produce per-episode history states.      Re

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): Roll the base dynamics forward ``horizon`` steps from a perturbed state.      Re

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): Least-squares affine map ``y ≈ x @ W.T + c``; returns (W, c).

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): Recover each hidden operator ``M_k`` from latent transitions and compare to trut

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): Predict held-out AB/BA compositions from single-operator estimates only.      Si

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): Explicit AB-vs-BA gap; the battery must be genuinely non-commutative.

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (1): Mean trajectory, operator-induced, and subject-transfer response-profile distanc

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (1): Evaluate the gate and return the dossier-schema payload.      All checks are con

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (1): Persist a gate payload as pretty, sorted JSON (reuses repro.write_json).

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (1): Load a gate payload written by :func:`write_evidence_gate`.

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (1): Narrow loaded YAML at the command boundary after load_config validation.

## Knowledge Gaps
- **159 isolated node(s):** `Tests for the KTM A100 evidence intake auditor (synthetic fixtures only).`, `Build a synthetic KTM A100 evidence bundle folder; returns the bundle root.`, `Base (no-ablation) failure-analysis report shape, finiteness, and gate disciplin`, `The ablation matrix loads and smoke-runs, and never earns recovery.`, `Each shipped ablation / Sprint 3C YAML loads into a valid KTMTrainConfig.` (+154 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 23`** (25 nodes): `ExpandedCliTests`, `.run_cli()`, `.run_script()`, `.test_bids_prepare_writes_event_manifest_when_derivative_exists()`, `.test_data_and_split_audits()`, `.test_estimate_and_train_dry_run()`, `.test_eval_classification_leakage_demo_warns_on_deprecated_typo_alias()`, `.test_eval_classification_leakage_demo_writes_paper_artifacts()`, `.test_eval_leakage_demo_subcommand_accepts_multi_seed()`, `.test_eval_nfc_synthetic_require_pass_fails_needs_evidence()`, `.test_eval_nfc_synthetic_suite_honors_multiple_seeds()`, `.test_eval_nfc_synthetic_suite_writes_artifacts()`, `.test_eval_rejects_options_before_real_subcommand()`, `.test_make_figures_treats_malformed_gate_as_absent()`, `.test_make_figures_treats_malformed_summary_as_absent()`, `.test_make_tables_treats_malformed_gate_as_plumbing()`, `.test_make_tables_treats_malformed_summary_as_plumbing()`, `.test_report_compare_surfaces_malformed_json_artifact()`, `.test_report_compare_writes_aggregate_artifacts()`, `.test_report_run_dir()`, `.test_report_run_dir_keeps_summary_claim_source_of_truth_with_valid_gate()`, `.test_report_run_dir_rejects_invalid_colocated_gate()`, `.test_report_run_dir_uses_colocated_prepared_baseline_suite()`, `_valid_paper_mode_gate()`, `test_expanded.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (18 nodes): `ArtifactDocsContractsTests`, `._run_docker_launcher_dry_run()`, `.test_a100_h100_configs_scripts_and_paper_docs_exist()`, `.test_a100_runbook_separates_fast_and_heavy_lanes()`, `.test_a100_slurm_scripts_require_safe_inputs()`, `.test_agent_deploy_docs_and_dockerfile_are_6gpu_first()`, `.test_chapman_first_run_launcher_contains_required_sequence()`, `.test_claims_doc_blocks_forbidden_claims()`, `.test_docker_6gpu_runner_contains_required_sequence()`, `.test_docker_launcher_default_and_diagnostic_cuda_visible_devices()`, `.test_docker_launcher_honors_explicit_container_cuda_visible_devices()`, `.test_docker_launcher_uses_container_local_cuda_visible_devices()`, `.test_moabb_benchmark_script_blocks_slurm_tmp_fallback()`, `.test_moabb_scripts_and_cluster_configs_use_benchmark_windows()`, `.test_operator_run_bundle_files_are_self_contained()`, `.test_runpod_rehearsal_is_budget_gated()`, `.test_tribe_style_does_not_become_required_dependency()`, `test_docs_contracts.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (5 nodes): `KtmHandoffTests`, `.setUp()`, `.test_evidence_bundle_includes_run_files_excludes_secrets()`, `.test_handoff_zip_shape_and_runner_checksum()`, `test_ktm_handoff.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (3 nodes): `Second-difference smoothness penalty for future EEG windows.`, `smoothness_loss()`, `metrics.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `Load a local NeuroTwin TRIBE-style config or seeded defaults.          No pretra`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Compatibility shim for TRIBE-style callers.          Prefer :meth:`from_checkpoi`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Evaluate the gate and return the dossier-schema payload.      All checks are con`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Persist a gate payload as pretty, sorted JSON (reuses repro.write_json).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Roll the autonomous base dynamics to produce per-episode history states.      Re`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Roll the base dynamics forward ``horizon`` steps from a perturbed state.      Re`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `Least-squares affine map ``y ≈ x @ W.T + c``; returns (W, c).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `Recover each hidden operator ``M_k`` from latent transitions and compare to trut`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `Predict held-out AB/BA compositions from single-operator estimates only.      Si`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `Explicit AB-vs-BA gap; the battery must be genuinely non-commutative.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `Mean trajectory, operator-induced, and subject-transfer response-profile distanc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (1 nodes): `Evaluate the gate and return the dossier-schema payload.      All checks are con`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (1 nodes): `Persist a gate payload as pretty, sorted JSON (reuses repro.write_json).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (1 nodes): `Load a gate payload written by :func:`write_evidence_gate`.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (1 nodes): `Narrow loaded YAML at the command boundary after load_config validation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Data schemas, split manifests, and leakage guards.` connect `Community 8` to `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 6`, `Community 7`, `Community 10`, `Community 13`, `Community 15`, `Community 24`?**
  _High betweenness centrality (0.085) - this node is a cross-community bridge._
- **Why does `NumpyRidgeBaseline` connect `Community 3` to `Community 1`, `Community 6`, `Community 7`, `Community 8`, `Community 20`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Why does `write_json()` connect `Community 9` to `Community 0`, `Community 1`, `Community 2`, `Community 4`, `Community 7`, `Community 11`, `Community 13`, `Community 15`, `Community 19`, `Community 20`, `Community 21`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **Are the 139 inferred relationships involving `ValueError` (e.g. with `_parse_seeds()` and `_require_bundle_rel()`) actually correct?**
  _`ValueError` has 139 INFERRED edges - model-reasoned connections that need verification._
- **Are the 91 inferred relationships involving `NumpyRidgeBaseline` (e.g. with `DualFieldTests` and `BaselinesAndArchitectureTests`) actually correct?**
  _`NumpyRidgeBaseline` has 91 INFERRED edges - model-reasoned connections that need verification._
- **Are the 52 inferred relationships involving `Data schemas, split manifests, and leakage guards.` (e.g. with `EMContext` and `IdleRecordingMetadata`) actually correct?**
  _`Data schemas, split manifests, and leakage guards.` has 52 INFERRED edges - model-reasoned connections that need verification._
- **Are the 59 inferred relationships involving `TorchMLPBaseline` (e.g. with `BaselinesAndArchitectureTests` and `RegressionTask`) actually correct?**
  _`TorchMLPBaseline` has 59 INFERRED edges - model-reasoned connections that need verification._