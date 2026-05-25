# Graph Report - Kahlus Vidya v1  (2026-05-24)

## Corpus Check
- 70 files · ~24,324 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 437 nodes · 892 edges · 20 communities detected
- Extraction: 61% EXTRACTED · 39% INFERRED · 0% AMBIGUOUS · INFERRED: 345 edges (avg confidence: 0.75)
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
- [[_COMMUNITY_Community 20|Community 20]]

## God Nodes (most connected - your core abstractions)
1. `run_prepared_training()` - 23 edges
2. `NeuralEventBatch` - 23 edges
3. `_cmd_train()` - 22 edges
4. `NeuralStateSpaceTranslator` - 22 edges
5. `build_split_manifest()` - 19 edges
6. `audit_prepared_eval_inputs()` - 18 edges
7. `_cmd_data_prepare()` - 16 edges
8. `_cmd_data_smoke()` - 16 edges
9. `make_synthetic_recordings()` - 16 edges
10. `RecordingRecord` - 15 edges

## Surprising Connections (you probably didn't know these)
- `BaselinesAndArchitectureTests` --uses--> `NeuralStateSpaceTranslator`  [INFERRED]
  tests/test_baselines_and_architecture.py → src/neurotwin/models/torch_models.py
- `ModelShapeTests` --uses--> `TinyTransformerBaseline`  [INFERRED]
  tests/test_model_shapes.py → src/neurotwin/models/torch_models.py
- `AdapterTests` --uses--> `MissingOptionalDependency`  [INFERRED]
  tests/test_adapters_moabb_bids.py → src/neurotwin/adapters/moabb.py
- `ModelMetadataGeometryTests` --uses--> `NeuralStateSpaceTranslator`  [INFERRED]
  tests/test_model_metadata_geometry.py → src/neurotwin/models/torch_models.py
- `WindowsAndMoabbLoaderTests` --uses--> `WindowSpec`  [INFERRED]
  tests/test_windows_and_moabb_loader.py → src/neurotwin/data/windows.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.07
Nodes (31): _cmd_estimate(), _cmd_split_audit(), _cmd_train(), _config_value(), _has_prepared_training_inputs(), ConfigError, load_config(), Raised when an experiment config cannot be loaded or validated. (+23 more)

### Community 1 - "Community 1"
Cohesion: 0.08
Nodes (26): _metrics(), bootstrap_ci(), mae(), mse(), pearsonr(), r2_score(), rank_models(), RankingRow (+18 more)

### Community 2 - "Community 2"
Cohesion: 0.1
Nodes (35): _events_for(), _infer_modality(), _is_bids_signal(), _listlike(), _load_timeseries_derivative(), _parse_entities(), _read_tsv(), _read_tsv_rows() (+27 more)

### Community 3 - "Community 3"
Cohesion: 0.09
Nodes (21): audit_split_manifest(), AuditReport, _forbidden_metadata(), _metadata_group_overlap(), _window_overlap(), check_manifest_leakage(), LeakageReport, Check record reuse and held-out group overlap across train/val/test. (+13 more)

### Community 4 - "Community 4"
Cohesion: 0.08
Nodes (18): _fit_neurotwin(), Run a tiny deterministic CPU training loop for CLI and CI smoke tests., run_synthetic_training(), TrainingSmokeResult, _latent(), make_synthetic_event_batches(), _noise(), _projection() (+10 more)

### Community 5 - "Community 5"
Cohesion: 0.1
Nodes (25): _cross_modal_task(), _fit_ridge(), _fit_torch_sequence_model(), _flatten_time(), _future_task(), _make_paired_windows(), _masked_reconstruction_task(), Run tiny local baselines on paired synthetic windows.      This is intentionally (+17 more)

### Community 6 - "Community 6"
Cohesion: 0.09
Nodes (23): audit_prepared_eval_inputs(), _prepared_windows_by_split(), PreparedEvalAuditReport, Audit prepared eval inputs before any benchmark score is trusted., _record_id(), _window_overlap_violations(), event_manifest_summary(), _jsonable() (+15 more)

### Community 7 - "Community 7"
Cohesion: 0.1
Nodes (15): cleanup_process_group(), DistributedInfo, get_distributed_info(), get_rank_metrics_path(), maybe_init_process_group(), unwrap_model(), wrap_ddp_if_initialized(), _all_synthetic() (+7 more)

### Community 8 - "Community 8"
Cohesion: 0.15
Nodes (24): format_prepared_eval_audit(), _cmd_data_smoke(), _cmd_eval(), build_prepared_window_tasks(), _cross_modal_task_from_windows(), _dataset_site_generalization_from_windows(), _first_modality_with_splits(), format_prepared_baseline_report() (+16 more)

### Community 9 - "Community 9"
Cohesion: 0.2
Nodes (7): dataset_registry(), DatasetAdapterSpec, RegistryTests, permissive_upstreams(), quarantined_upstreams(), upstream_registry(), UpstreamSpec

### Community 10 - "Community 10"
Cohesion: 0.19
Nodes (8): _cmd_report(), competitor_registry(), CompetitorSpec, generate_run_report(), generate_suite_report(), default_translation_tasks(), TaskSpec, TaskSpecTests

### Community 11 - "Community 11"
Cohesion: 0.22
Nodes (1): Data schemas, split manifests, and leakage guards.

### Community 12 - "Community 12"
Cohesion: 0.33
Nodes (6): _cmd_doctor(), _check_runs_writable(), DoctorCheck, DoctorReport, format_doctor_report(), run_doctor()

### Community 13 - "Community 13"
Cohesion: 0.48
Nodes (1): ExpandedCliTests

### Community 14 - "Community 14"
Cohesion: 0.5
Nodes (1): ResearchArtifactTests

### Community 15 - "Community 15"
Cohesion: 0.5
Nodes (2): NeuralStateSpaceTranslatorConfig, Configuration shell for the future NeuroTwin model implementation.

### Community 16 - "Community 16"
Cohesion: 0.67
Nodes (1): CliReportTests

### Community 17 - "Community 17"
Cohesion: 1.0
Nodes (2): _clone_or_checkout(), main()

### Community 18 - "Community 18"
Cohesion: 1.0
Nodes (2): _flatten_metrics(), main()

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (1): Encode observed modalities into shared latent neural tokens.

## Knowledge Gaps
- **13 isolated node(s):** `Raised when an experiment config cannot be loaded or validated.`, `Closed-form ridge baseline for sanity checks and tiny CPU benchmarks.`, `Per-timepoint MLP baseline for neural windows.`, `Small Conv1D/TCN-style baseline with stable [batch, time, feature] IO.`, `Configuration shell for the future NeuroTwin model implementation.` (+8 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 11`** (9 nodes): `Data schemas, split manifests, and leakage guards.`, `__init__.py`, `__init__.py`, `__init__.py`, `__init__.py`, `__init__.py`, `__init__.py`, `__init__.py`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 13`** (7 nodes): `ExpandedCliTests`, `.run_cli()`, `.test_bids_prepare_writes_event_manifest_when_derivative_exists()`, `.test_data_and_split_audits()`, `.test_estimate_and_train_dry_run()`, `.test_report_run_dir()`, `test_cli_expanded.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 14`** (4 nodes): `ResearchArtifactTests`, `.test_claims_doc_blocks_forbidden_claims()`, `.test_h100_configs_scripts_and_paper_docs_exist()`, `test_research_artifacts.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 15`** (4 nodes): `translator.py`, `NeuralStateSpaceTranslatorConfig`, `.describe()`, `Configuration shell for the future NeuroTwin model implementation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 16`** (3 nodes): `CliReportTests`, `.test_report_mentions_corrected_boss_fight_and_split_rules()`, `test_cli_report.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 17`** (3 nodes): `vendor_upstreams.py`, `_clone_or_checkout()`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 18`** (3 nodes): `_flatten_metrics()`, `main()`, `make_tables.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `Encode observed modalities into shared latent neural tokens.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_cmd_train()` connect `Community 0` to `Community 3`, `Community 4`, `Community 6`, `Community 7`?**
  _High betweenness centrality (0.150) - this node is a cross-community bridge._
- **Why does `run_prepared_training()` connect `Community 7` to `Community 0`, `Community 1`, `Community 3`, `Community 4`, `Community 6`, `Community 8`?**
  _High betweenness centrality (0.140) - this node is a cross-community bridge._
- **Why does `PreparedSuiteConfig` connect `Community 8` to `Community 1`, `Community 3`, `Community 5`, `Community 6`, `Community 7`?**
  _High betweenness centrality (0.129) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `run_prepared_training()` (e.g. with `.test_prepared_training_writes_checkpoint_and_metrics()` and `_cmd_train()`) actually correct?**
  _`run_prepared_training()` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `NeuralEventBatch` (e.g. with `NeuralEventBatchTests` and `Create paired synthetic recordings that mirror the recording manifest.`) actually correct?**
  _`NeuralEventBatch` has 20 INFERRED edges - model-reasoned connections that need verification._
- **Are the 19 inferred relationships involving `_cmd_train()` (e.g. with `load_config()` and `estimate_config()`) actually correct?**
  _`_cmd_train()` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `NeuralStateSpaceTranslator` (e.g. with `ModelMetadataGeometryTests` and `BaselinesAndArchitectureTests`) actually correct?**
  _`NeuralStateSpaceTranslator` has 15 INFERRED edges - model-reasoned connections that need verification._