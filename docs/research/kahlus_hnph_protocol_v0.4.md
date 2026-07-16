# Kahlus HNPH Protocol v0.4

**Status:** frozen protocol; source qualification pending; no empirical HNPH claim authorized.

Protocol v0.4 supersedes v0.3 for all future claim-mode work. Version 0.3 remains immutable historical provenance. The machine-readable contract is `configs/protocol/hnph_phase0_v0.4.yaml`.

## Scientific endpoint

HNPH is an oracle-conditioned sleep-transition predictability protocol. At an issue time, the evaluator provides the current operational sleep-stage annotation and causal stage history. A model assigns probabilities to a five-way future annotation outcome (`no_event`, `Wake`, `NREM`, `REM`, `Ambiguous`) in preregistered lead bands. This is not a biological transition detector, deployment study, consciousness measure, clinical system, or mechanism claim.

The primary estimand is subject-balanced incremental categorical log skill, in bits per anchor, over the best eligible nuisance comparator. Log score is a proper score. Its ideal conditional-information interpretation requires oracle conditional distributions and is not asserted for fitted finite-sample models.

## Cohort contract

The intended development cohort is DOD-H. DOD-O is the sealed external acquisition/population cohort. This assignment becomes active only after a fail-closed local qualification establishes, for each dataset:

- five separately identifiable independent rater streams;
- stable subject identities suitable for person-grouped splitting;
- physical channel units and sampling rates;
- dataset version, license identifier, immutable source hash, and per-rater annotation hashes.

Until all checks pass, `migration_authorized` and `claim_mode_authorized` remain false. If the downloadable source lacks individual annotations, migration stops. Consensus labels are not a substitute. Sleep-EDF and CAP may be used only for explicitly descriptive single-label transport checks and cannot satisfy the construct-validity gate.

## Label target

For target rater \(r\), the soft target is built only from the other raters. The target rater must never contribute to its own consensus. At least three independent consensus raters are required at every scored anchor; v0.4 additionally requires five independent source streams during qualification. Every target artifact binds the target-rater ID, contributing-rater IDs, annotation hashes, stage mapping, and output hash.

## Leakage and inference contract

- Split by person before fitting any transform, imputation, calibration, baseline, or model.
- Input and target supports are disjoint in physical time, with the causal guard recorded.
- Forbidden identifiers and acquisition proxies are excluded or explicitly audited.
- DOD-O remains unopened until protocol and model-family freeze.
- Inference is subject-balanced with a subject-cluster max-t bootstrap using at least 2,000 replicates.
- The model must beat the best eligible baseline and pass label shuffle, time shift, cluster permutation, missingness, nuisance-probe, synthetic-known, and synthetic-null controls.
- Power requirements are at least 12 independent subject clusters, 8 event subjects, 100 positive primary-band anchors, and 80% target power under frozen design assumptions.

## Evidence contract

The runner must emit paired JSON and Markdown evidence bound to the protocol hash and all source, split, target, transform, baseline, control, inference, and qualification hashes. A valid null or blocked study is an acceptable outcome. A positive frontier requires all gates and cannot be inferred from a schematic, development result, isolated lead band, or uncalibrated score.

The external result remains sealed during manuscript and figure development. The v0.4 figure system therefore labels unfinished stages `UNRUN` and refuses empirical-looking result panels without claim-eligible evidence.

## Boundary for later protocols

NFC, geometry-aware operators, future acquisition hardware, disease applications, and passive-PCI hypotheses are outside v0.4. Any such work requires a separately frozen protocol and evidence gate.
