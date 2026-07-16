# HNPH Figure Provenance

- protocol: `kahlus.hnph.phase0.v0.4`
- protocol_sha256: `401a8e47db3aefc5c549fec956254f291f71e43c6bf636c464551df552acc839`
- claim_scope: `protocol_only_no_empirical_hnph_frontier_claim`
- source_qualification_status: `unverified`
- external_test_opened: `false`
- command: `python scripts/analysis/plot_hnph_preprint_figures.py --protocol configs/protocol/hnph_phase0_v0.4.yaml --out-dir docs/figures/hnph_protocol`

Raw data and local paths are not included. The Sleep-EDF appendix panel is a hash-bound descriptive source asset, not independently qualified HNPH v0.4 evidence.

## Dataset versions

- DOD-H: `unverified`
- DOD-O: `unverified`

## Input hashes

- protocol: `401a8e47db3aefc5c549fec956254f291f71e43c6bf636c464551df552acc839`
- renderer: `b981c2451049ebcdc5f86a734229461293a360923005ed93ad8f89db1516ee30`
- sleep_edf_descriptive_pdf: `01bf87fae4b2ffad236f7987c2549f47827f07b89b5e2c075cc574a7a16b1855`
- sleep_edf_descriptive_png: `92d27ff814c959c58ddd68b5499225f3d84e48f5247bae32ff0444110c710c87`
- sleep_edf_descriptive_provenance: `705895d62a035ab13302a67c7695201f32739356bd9be20fd7a950dd40d46110`

## Figures

### fig1_operational_task

- classification: `raw_illustration`
- status: `raw_illustration`
- PDF SHA-256: `4586757ddb9da695a1223100d22fdd0ca7f7d1e04eb643a46a6a8ad164c7031c`
- PNG SHA-256: `d3b5aece90e67b4b5dd4ffec442d7927c408831642d4012d044b535556c69276`
- panels: causal task, lead bands, score comparison
- caption: Operational HNPH task. An oracle current-stage annotation and causal EEG history precede a guard interval and four future lead bands. The five-way leave-one-rater-out target is scored against the best eligible nuisance comparator. This is a task schematic, not a performance result.

### fig2_leakage_label_contract

- classification: `conceptual`
- status: `conceptual`
- PDF SHA-256: `ec31a0d19f8be603ef3e12c4afdcbfe389127cc62b349381698b322d0890dd48`
- PNG SHA-256: `61d6315fb399bb851fe834c8a0599755c4837648a94871d27ad6edc0dc2897fc`
- panels: split, firebreak, rater exclusion
- caption: Leakage and label contract. People are split before any fitted transform; input and outcome supports are disjoint in physical time; and the held-out target rater never contributes to the consensus target.

### fig3_study_flow

- classification: `conceptual`
- status: `conceptual`
- PDF SHA-256: `d8d20ed9c258c341f08eeb9de2989e3c5867e87508062fefc1166755ec3e7d74`
- PNG SHA-256: `f7e62a78c5f2163b9146cdf7d75c7d1784d2034c8239c377e1e1fd10e3ccabe0`
- panels: development, gates, sealed external
- caption: Baseline-first study flow. DOD-H source qualification and development gates precede model-family freeze; DOD-O remains sealed until then. Every stage is UNRUN in this protocol release.

### fig4_protocol_outcomes

- classification: `conceptual`
- status: `conceptual`
- PDF SHA-256: `54a91406739fbd5ae254073847b1145196d650bb60e35ffc0f18bc60cee6aa07`
- PNG SHA-256: `f3f23a062a238eaf4419df158b4204edde756687acfeaec3948add91559f37ce`
- panels: supported, null, failed, invalid
- caption: Possible protocol outcomes. A supported frontier, bounded or null result, failed calibration or control, and invalid execution are distinct outcomes. Symbolic bands encode decisions, not observed effects.

### figA1_verified_example

- classification: `raw_illustration`
- status: `descriptive_single_label_transport_not_claim_evidence`
- PDF SHA-256: `01bf87fae4b2ffad236f7987c2549f47827f07b89b5e2c075cc574a7a16b1855`
- PNG SHA-256: `92d27ff814c959c58ddd68b5499225f3d84e48f5247bae32ff0444110c710c87`
- panels: trace, spectrogram, hypnogram
- caption: Descriptive Sleep-EDF trace, spectrogram, and single-label hypnogram from record SC4002E0. The origin package reports 100 Hz sampling and hash-bound PSG/hypnogram sources. This transport illustration is not independently source-qualified under HNPH v0.4 and cannot enable a repeated-rater construct-validity or empirical frontier claim.

### figA2_lineage

- classification: `conceptual`
- status: `conceptual`
- PDF SHA-256: `8f0cd1cd57aaf1502c49672aa843204fc5305d80d7adde331c4f410a4494f636`
- PNG SHA-256: `55792167a7f7a5dac4443f42f2f40ae1fac21afaba9057d6e414bc499cd2f170`
- panels: source, manifests, evidence
- caption: Raw-to-evidence lineage. Hash-bound local sources feed qualification, person-first manifests, causal targets, frozen comparisons, and paired JSON/Markdown evidence; raw neural samples remain outside git.

### figA3_score_ceiling

- classification: `conceptual`
- status: `conceptual`
- PDF SHA-256: `f8adbb23be3c70008f7dd8fd55f37df448815fbb0818d1cdf8cd88c21d2b0f96`
- PNG SHA-256: `ed28c6c740bbf4aed8ff15fe9b0e565355cb4967c4373512ad2a661ea15ac1a7`
- panels: proper score, oracle identity, data processing
- caption: Proper-score decomposition and observation-contraction ceiling. The conditional-information identity is an oracle statement, while contraction is assumption-dependent; neither is presented as a novel theorem.
