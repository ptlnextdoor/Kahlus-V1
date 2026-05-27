# Graph Report - /Users/aayu/conductor/workspaces/kahlus-v1/abuja  (2026-05-26)

## Corpus Check
- 70 files · ~70,846 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 518 nodes · 1085 edges · 29 communities detected
- Extraction: 62% EXTRACTED · 38% INFERRED · 0% AMBIGUOUS · INFERRED: 414 edges (avg confidence: 0.74)
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

## God Nodes (most connected - your core abstractions)
1. `run_prepared_training()` - 28 edges
2. `NeuralStateSpaceTranslator` - 28 edges
3. `NeuralEventBatch` - 28 edges
4. `_cmd_train()` - 22 edges
5. `audit_prepared_eval_inputs()` - 20 edges
6. `build_split_manifest()` - 19 edges
7. `RecordingRecord` - 18 edges
8. `_cmd_data_prepare()` - 17 edges
9. `_cmd_data_smoke()` - 17 edges
10. `make_synthetic_recordings()` - 16 edges

## Surprising Connections (you probably didn't know these)
- `NeuralEventBatch` --uses--> `Load a small MOABB dataset through its paradigm API when optional deps exist.`  [INFERRED]
  /Users/aayu/conductor/workspaces/kahlus-v1/abuja/src/neurotwin/data/schemas.py → src/neurotwin/adapters/moabb.py
- `NeuralEventBatch` --uses--> `Normalize MOABB/MNE trial arrays to NeuroTwin [time, channel] layout.`  [INFERRED]
  /Users/aayu/conductor/workspaces/kahlus-v1/abuja/src/neurotwin/data/schemas.py → src/neurotwin/adapters/moabb.py
- `NeuralEventBatch` --uses--> `Normalize MOABB/MNE trial arrays to NeuroTwin [time, channel] layout.`  [INFERRED]
  /Users/aayu/conductor/workspaces/kahlus-v1/abuja/src/neurotwin/data/schemas.py → src/neurotwin/adapters/moabb.py
- `Run a tiny deterministic CPU training loop for CLI and CI smoke tests.` --uses--> `NeuralStateSpaceTranslator`  [INFERRED]
  src/neurotwin/training/smoke.py → /Users/aayu/conductor/workspaces/kahlus-v1/abuja/src/neurotwin/models/torch_models.py
- `NeuralStateSpaceTranslator` --uses--> `Run tiny local baselines on paired synthetic windows.      This is intentionally`  [INFERRED]
  /Users/aayu/conductor/workspaces/kahlus-v1/abuja/src/neurotwin/models/torch_models.py → src/neurotwin/benchmarks/baseline_suite.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.08
Nodes (35): _baseline_catalog(), BaselineFailure, _cross_modal_task(), _fit_ridge(), _fit_torch_sequence_model(), _flatten_time(), _future_task(), _make_paired_windows() (+27 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (30): audit_prepared_eval_inputs(), _prepared_windows_by_split(), PreparedEvalAuditReport, Audit prepared eval inputs before any benchmark score is trusted., Audit prepared eval inputs before any benchmark score is trusted., _record_id(), _window_overlap_violations(), event_manifest_summary() (+22 more)

### Community 2 - "Community 2"
Cohesion: 0.08
Nodes (33): format_prepared_eval_audit(), _metrics(), _cmd_eval(), bandpower_error(), bootstrap_ci(), mae(), mse(), pearsonr() (+25 more)

### Community 3 - "Community 3"
Cohesion: 0.08
Nodes (30): audit_split_manifest(), AuditReport, _forbidden_metadata(), _metadata_group_overlap(), _window_overlap(), _cmd_split_audit(), check_manifest_leakage(), LeakageReport (+22 more)

### Community 4 - "Community 4"
Cohesion: 0.07
Nodes (17): _fit_neurotwin(), Run a tiny deterministic CPU training loop for CLI and CI smoke tests., run_synthetic_training(), TrainingSmokeResult, ModelMetadataGeometryTests, ModelShapeTests, TrainingSmokeTests, _build_backbone() (+9 more)

### Community 5 - "Community 5"
Cohesion: 0.08
Nodes (26): _cmd_doctor(), _cmd_estimate(), _cmd_train(), _config_value(), _has_prepared_training_inputs(), ConfigError, load_config(), Raised when an experiment config cannot be loaded or validated. (+18 more)

### Community 6 - "Community 6"
Cohesion: 0.09
Nodes (22): cleanup_process_group(), DistributedInfo, get_distributed_info(), get_rank_metrics_path(), maybe_init_process_group(), unwrap_model(), wrap_ddp_if_initialized(), _aggregate_task_results() (+14 more)

### Community 7 - "Community 7"
Cohesion: 0.14
Nodes (24): _cmd_data_prepare(), _cmd_data_smoke(), save_data_manifest(), save_leakage_report(), save_split_manifest(), balanced_trial_subset(), _build_moabb_paradigm(), _build_record_id() (+16 more)

### Community 8 - "Community 8"
Cohesion: 0.13
Nodes (21): _cmd_report(), competitor_registry(), CompetitorSpec, _adaptation_rows(), _csv_cell(), _csv_rows(), _flatten_metrics(), generate_compare_report() (+13 more)

### Community 9 - "Community 9"
Cohesion: 0.18
Nodes (20): build_prepared_window_tasks(), _cross_modal_task_from_windows(), _dataset_site_generalization_from_windows(), _first_modality_with_splits(), _future_task_from_windows(), _future_xy(), _group_windows(), _masked_task_from_windows() (+12 more)

### Community 10 - "Community 10"
Cohesion: 0.17
Nodes (18): bids_manifest_summary(), _events_for(), _infer_modality(), _is_bids_signal(), _listlike(), _load_timeseries_derivative(), _parse_entities(), _read_tsv() (+10 more)

### Community 11 - "Community 11"
Cohesion: 0.2
Nodes (7): dataset_registry(), DatasetAdapterSpec, RegistryTests, permissive_upstreams(), quarantined_upstreams(), upstream_registry(), UpstreamSpec

### Community 12 - "Community 12"
Cohesion: 0.22
Nodes (1): Data schemas, split manifests, and leakage guards.

### Community 13 - "Community 13"
Cohesion: 0.28
Nodes (1): MoabbCliAndGeneralizationTests

### Community 14 - "Community 14"
Cohesion: 0.43
Nodes (1): ExpandedCliTests

### Community 15 - "Community 15"
Cohesion: 0.33
Nodes (1): ResearchArtifactTests

### Community 16 - "Community 16"
Cohesion: 0.53
Nodes (1): ManifestAuditAndTorchrunTests

### Community 17 - "Community 17"
Cohesion: 0.83
Nodes (3): _flatten_metrics(), main(), _read_summary()

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
Nodes (1): Per-timepoint MLP baseline for neural windows.

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (1): Small Conv1D/TCN-style baseline with stable [batch, time, feature] IO.

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (1): Small Transformer baseline for CPU shape and smoke tests.

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): CPU debug stand-in for long-sequence SSM baselines until Mamba is pinned.

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): Modality-tokenizer + shared latent dynamics + modality-readout scaffold.

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): Encode observed modalities into shared latent neural tokens.

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (1): Encode observed modalities into shared latent neural tokens.

## Knowledge Gaps
- **20 isolated node(s):** `Raised when an experiment config cannot be loaded or validated.`, `Closed-form ridge baseline for sanity checks and tiny CPU benchmarks.`, `Per-timepoint MLP baseline for neural windows.`, `Small Conv1D/TCN-style baseline with stable [batch, time, feature] IO.`, `Configuration shell for the future NeuroTwin model implementation.` (+15 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 21`** (2 nodes): `main()`, `make_figures.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `Per-timepoint MLP baseline for neural windows.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `Small Conv1D/TCN-style baseline with stable [batch, time, feature] IO.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `Small Transformer baseline for CPU shape and smoke tests.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `CPU debug stand-in for long-sequence SSM baselines until Mamba is pinned.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `Modality-tokenizer + shared latent dynamics + modality-readout scaffold.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `Encode observed modalities into shared latent neural tokens.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `Encode observed modalities into shared latent neural tokens.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_prepared_training()` connect `Community 6` to `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 9`?**
  _High betweenness centrality (0.154) - this node is a cross-community bridge._
- **Why does `_cmd_train()` connect `Community 5` to `Community 1`, `Community 3`, `Community 4`, `Community 6`, `Community 7`?**
  _High betweenness centrality (0.134) - this node is a cross-community bridge._
- **Why does `PreparedSuiteConfig` connect `Community 9` to `Community 0`, `Community 1`, `Community 2`, `Community 3`, `Community 6`, `Community 7`?**
  _High betweenness centrality (0.129) - this node is a cross-community bridge._
- **Are the 18 inferred relationships involving `run_prepared_training()` (e.g. with `.test_prepared_training_writes_checkpoint_and_metrics()` and `.test_prepared_training_runs_all_neural_translation_tasks()`) actually correct?**
  _`run_prepared_training()` has 18 INFERRED edges - model-reasoned connections that need verification._
- **Are the 21 inferred relationships involving `NeuralStateSpaceTranslator` (e.g. with `ModelMetadataGeometryTests` and `BaselinesAndArchitectureTests`) actually correct?**
  _`NeuralStateSpaceTranslator` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `NeuralEventBatch` (e.g. with `NeuralEventBatchTests` and `Create paired synthetic recordings that mirror the recording manifest.`) actually correct?**
  _`NeuralEventBatch` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Are the 19 inferred relationships involving `_cmd_train()` (e.g. with `load_config()` and `estimate_config()`) actually correct?**
  _`_cmd_train()` has 19 INFERRED edges - model-reasoned connections that need verification._
