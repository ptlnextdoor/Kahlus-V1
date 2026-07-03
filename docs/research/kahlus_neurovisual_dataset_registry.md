# Kahlus NV-1 Neurovisual Dataset Registry

NV-1 is a Kahlus side branch for dataset discovery and metadata discipline. It does not replace the Kahlus v1 EEG paper lane or the ResearchDock device lane.

This registry is metadata-only. It is not a diagnosis and performs no bulk dataset download, no A100 run, no cluster job, no private patient data use, and no clinical diagnosis.

## Registry Schema

Each entry includes `dataset_name`, `source_url_or_identifier`, `verification_status`, `verification_source`, `metadata_source_url`, `date_checked`, `modality`, `task`, `population`, `labels_available`, `event_annotations_available`, `subjective_symptom_annotations_available`, `eeg_montage_and_channels`, `sampling_rate`, `file_format`, `license_and_access_requirements`, `download_size`, `split_strategy_feasibility`, `leakage_risks`, `neurovisual_relevance_score`, `adapter_priority`, and `notes`.

`metadata_source_url` is a machine-readable source pointer for confirmed entries. Unverified leads use `UNVERIFIED`; rejected private-data entries use `REJECTED`. Do not invent accession URLs for OpenNeuro/NEMAR leads.

## Score Definitions

- 0: no visual, seizure, perceptual, or relevant EEG content.
- 1: incidental visual task or generic EEG during visual stimulation.
- 2: EEG with seizure/event annotations, visual ERP, photic stimulation, or perceptual-task structure, but no subjective symptom labels.
- 3: EEG/event data paired with subjective visual symptom labels, aura descriptions, clinician-coded neurovisual episode phenotype, or explicit symptom-event alignment.

Adapter priority and neurovisual relevance are separate fields. HBN-EEG can be adapter priority 1 while scoring only 2 for neurovisual specificity.

## Confirmed Dataset Anchors

- HBN-EEG / EEG Foundation Challenge: confirmed seed metadata from `https://arxiv.org/abs/2506.19141`; adapter priority 1; neurovisual relevance score 2.
- CHB-MIT Scalp EEG Database: confirmed seed metadata from `https://physionet.org/content/chbmit/1.0.0/`; adapter priority 2; neurovisual relevance score 2.
- TUSZ / TUH EEG Seizure Corpus: confirmed seed metadata from `https://isip.piconepress.com/projects/nedc/html/tuh_eeg/`; adapter priority 3; neurovisual relevance score 2.

None of these receive score 3 in NV-1 because the seeded metadata does not verify subjective visual symptom labels, aura descriptions, clinician-coded neurovisual episode phenotype, or explicit symptom-event alignment.

## Unverified Leads

OpenNeuro and NEMAR leads for EEG plus pupillometry plus PPG working-memory tasks, EEG plus BDI reward/probabilistic selection tasks, photic stimulation, visual ERP, perceptual episode EEG, migraine, or visual aura remain unverified until accession number, subject count, task description, modalities, license, and access terms are confirmed.

No OpenNeuro ID is hardcoded as confirmed in this sprint.

## Verification Summary Artifact

The registry builder writes `neurovisual_registry_verification_summary.json` alongside the registry. It records dataset sources searched, `metadata_source_urls`, confirmed/unverified/rejected counts, OpenNeuro verification results, score-3 assignments, rejected datasets, top adapter priorities, and explicit `bulk_dataset_download=false` and `a100_jobs_launched=false` execution flags.

## Metadata-Only Adapter Plan Artifact

The registry builder also writes `neurovisual_adapter_plan.json`. This is planning metadata only: it records local manifest fields, split/audit strategy, and leakage risks for HBN-EEG, CHB-MIT, and TUSZ. It does not implement adapters, download data, run baselines, run models, launch A100, or make clinical claims.

## Local Manifest Contract

Future adapters must begin from a user-provided local manifest, not hardcoded paths or downloaded raw data. The local manifest requires `record_id`, `subject_id`, `session_id`, `dataset_name`, `signal_path`, `sampling_rate`, `channel_names`, `event_annotations_path`, `task_label`, and `license_or_access_confirmation`.

`validate_local_manifest_records()` checks required fields, positive sampling rate, non-empty channel names, and that each `dataset_name` is a confirmed registry entry. It rejects unverified OpenNeuro/NEMAR leads and rejected private-data targets. It does not check file existence or execute adapters in NV-1 because no real local dataset paths are provided in this sprint.

The registry builder writes `neurovisual_local_manifest_schema.json` with required fields, field type expectations, confirmed allowed dataset names, and blocked unverified or rejected dataset names. This is a schema artifact only; it does not include fake dataset paths or authorize adapters.

Run `scripts/audit_neurovisual_local_manifest.py --manifest <manifest.json> --registry <neurovisual_dataset_registry.json>` to audit an explicit manifest file. The CLI returns nonzero for unverified, rejected, unknown, incomplete, or malformed records and still does not download data or execute adapters.

## Split Audit Plan Artifact

The registry builder writes `neurovisual_split_audit_plan.json` with required split keys, leakage checks, and gates that must pass before any model work. The plan requires subject/session/record split discipline and a baseline ladder before models, but it does not execute splits, run baselines, run models, download data, or launch A100.

## Local Split Audit Validator

`validate_local_split_records()` audits explicit local manifest records with an added `split_name` field. `split_name` belongs to the split-audit layer, not the base local manifest contract, because the manifest describes available records while the split audit describes assignment.

Allowed split names are `train`, `validation`, and `test`. The validator first applies `validate_local_manifest_records()`, then rejects subject, session, record, signal path, or event annotation path overlap across different splits. It returns `split_audit_executed=true` while still recording `adapters_implemented=false`, `baselines_run=false`, `models_run=false`, `bulk_dataset_download=false`, `a100_jobs_launched=false`, and `raw_file_existence_checked=false`.

Run `scripts/audit_neurovisual_local_split.py --manifest <split_manifest.json> --registry <neurovisual_dataset_registry.json>` to audit an explicit split manifest. The CLI prints JSON, returns nonzero for invalid manifests or leakage, and still does not check raw files, download data, execute adapters, run baselines/models, or launch A100.

The registry builder also writes `neurovisual_synthetic_split_manifest.json` and `neurovisual_synthetic_split_audit.json`. These are stable handoff fixtures for the split-audit CLI and use only symbolic `USER_PROVIDED_*` path labels, not invented local files or raw dataset paths.

Run `scripts/run_neurovisual_fixture_replay.py --out-dir <out-dir>` to rebuild the registry package, replay `audit_neurovisual_local_split.py` against the generated synthetic split manifest, and write `neurovisual_fixture_replay_evidence.json`. This is a local handoff smoke only; it does not check raw files, download data, execute adapters, run baselines/models, or launch A100.

Run `scripts/build_neurovisual_handoff_manifest.py --registry-package-dir <registry-package-dir> --fixture-replay-evidence <neurovisual_fixture_replay_evidence.json> --out <handoff.json>` to summarize registry and fixture replay evidence under the existing `dataset_registry_ready` claim gate. The handoff manifest is for downstream adapter planning only and does not authorize raw-file access, adapters, baselines, models, clinical claims, or A100 execution.

Run `scripts/audit_neurovisual_handoff_manifest.py --handoff <handoff.json>` to verify the handoff manifest, claim gate, execution boundary flags, and input-artifact checksums before downstream adapter work. The audit returns nonzero for tampered or missing inputs and still does not check raw files, download data, execute adapters, run baselines/models, or launch A100.

Run `scripts/run_neurovisual_local_evidence_gate.py --out-dir <out-dir>` for a one-command local evidence gate. It runs fixture replay, builds the handoff manifest, audits the handoff, and writes `neurovisual_local_evidence_gate.json` plus `neurovisual_local_evidence_gate.md`; passing this gate means the local metadata/handoff artifacts are internally consistent, not that adapters, baselines, models, clinical claims, or A100 execution are approved.

Run `scripts/package_neurovisual_local_evidence_bundle.py --out-dir <out-dir>` to run the local evidence gate and package generated JSON/Markdown evidence into `neurovisual_local_evidence_bundle.tar.gz` with `neurovisual_local_evidence_bundle_manifest.json`. The bundle excludes raw neural data and checkpoints and remains a local handoff package, not approval for adapters, baselines, models, clinical claims, or A100 execution.

Run `scripts/audit_neurovisual_requirement_coverage.py --out-dir <out-dir>` to regenerate the local intake smoke and registry package, then write `neurovisual_requirement_coverage_audit.json` plus Markdown showing which NV-1 prompt requirements are covered by concrete evidence. This remains a metadata/intake coverage audit only; it does not download data, execute adapters, run baselines/models, approve clinical claims, or launch A100.

## Metadata Query Plan Artifact

The registry builder writes `neurovisual_metadata_query_plan.json` with planned OpenNeuro, NEMAR, MOABB, EEGDash, and public multimodal EEG catalog searches. This artifact records search terms and required verification fields only; it does not execute metadata queries, confirm accessions, download data, or authorize adapters.

## Registry Claim Gate Artifact

The registry builder writes `neurovisual_registry_claim_gate.json` and evaluates the registry, verification summary, adapter plan, and generated report under the `dataset_registry_ready` scope. The builder exits nonzero if the generated package contains a blocked NV-1 claim.

## Evidence Manifest Artifact

The registry builder writes `neurovisual_registry_evidence_manifest.json` with SHA-256 checksums and byte sizes for the registry JSON, verification summary, adapter plan, metadata query plan, local manifest schema, split audit plan, synthetic split fixture, claim gate, and Markdown report. The manifest records `metadata_only=true`, `bulk_dataset_download=false`, `a100_jobs_launched=false`, and `cluster_jobs_launched=false`.

## Registry Bundle Audit

Run `scripts/audit_neurovisual_registry_bundle.py --artifact-dir <registry-output-dir>` after building the registry package. The audit recomputes evidence-manifest checksums, verifies claim-gate status, and checks that download, A100, cluster, adapter, and metadata-query execution flags remain disabled.

## Rejected Datasets

Private or unprovided personal medical notes are rejected for NV-1. They are not used as rows, examples, fixtures, demo outputs, or adapter targets.

## Future Adapter Plan

1. Verify metadata source and access terms.
2. Add a local manifest adapter only after metadata is confirmed.
3. Run subject/session/event split and leakage audits.
4. Run baseline ladder before any model claim.
5. Keep claims research-only and non-diagnostic.

## No-Download Boundary

The registry builder records metadata and seed sources only. It does not download CHB-MIT, TUSZ, HBN, OpenNeuro, NEMAR, MOABB, EEGDash, or any large public dataset.
