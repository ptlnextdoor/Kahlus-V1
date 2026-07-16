# HNPH Canonical Implementation Ledger

**C0 canonical protocol:** `kahlus.hnph.phase0.v0.2`
**Active protocol:** `kahlus.hnph.phase0.v0.4`
**Ledger date:** 2026-07-13
**Current authorization:** protocol and source-qualification implementation
only; dataset migration, training, and claim-mode evaluation remain blocked

This ledger separates a frozen decision from an implemented and validated
capability. A checked documentation row does not authorize training or a result
claim.

## C0 Canonical Artifacts

| Artifact | Canonical responsibility | State |
| --- | --- | --- |
| `docs/research/kahlus_hnph_protocol_v0.2.md` | Concise controlling scientific and claim contract | Frozen in C0 |
| `configs/protocol/hnph_phase0_v0.2.yaml` | Machine-readable Phase 0 freeze | Frozen in C0 |
| `docs/audit/INVALIDATED_RESULT_REGISTRY.md` | Human-readable historical result disposition | Active in C0 |
| `docs/audit/invalidated_result_registry.json` | Machine-readable invalidation facts | Active in C0 |
| `docs/audit/DATA_MIGRATION_AND_QUARANTINE.md` | Retention, migration, and quarantine rules | Active in C0 |
| `docs/audit/REMEDIATION_ROADMAP.md` | Sequencing and launch authorization | Updated in C0 |

## v0.3 B2 Preregistration Additions

v0.3 preserves the v0.2 C0 freeze and adds claim-mode prerequisites before any
held-subject or CAP outcome is used for a claim. It does not invalidate a prior
claim-mode result because none has been produced.

| Artifact | Canonical responsibility | State |
| --- | --- | --- |
| `docs/research/kahlus_hnph_protocol_v0.3.md` | Concise v0.3 authority boundary and claim contract | Frozen before claim mode |
| `docs/research/hnph_b2_preregistration_addendum.md` | B2 design sensitivity, comparator, label-validity, and replication rules | Frozen before claim mode |
| `configs/protocol/hnph_phase0_v0.3.yaml` | Machine-readable v0.3 B2 preregistration | Frozen before claim mode |

The added B2 gates are: `epsilon = 0.02` bits/anchor as a design-sensitivity
floor; a validated semi-Markov comparator with a finite nuisance-challenger
audit; a five-way `no_event/Wake/NREM/REM/Ambiguous` operational target;
leave-one-rater-out label-reproducibility evidence; and a post-B2 unaffiliated
replication milestone. The comparator audit is not an asserted upper bound on
real-data misspecification. The B2 gate opens and hash-binds typed runner
artifacts, binds the canonical v0.3 protocol SHA-256 to the addendum, and does
not accept caller flags or hash-shaped strings as H3 authorization.

## v0.4 Source-Qualification Supersession

v0.4 preserves v0.3 as immutable history and supersedes it for all future
claim-mode work. The intended cohorts are DOD-H for development and sealed DOD-O
for external acquisition/population evaluation, conditional on local verification
of five separate rater streams, person identities, physical metadata, licensing,
and immutable source/annotation hashes. Source qualification is currently
`unverified`; migration and training are therefore unauthorized. Sleep-EDF and
CAP are descriptive single-label transport checks only under v0.4.

| Artifact | Canonical responsibility | State |
| --- | --- | --- |
| `docs/research/kahlus_hnph_protocol_v0.4.md` | Human-readable source, target, leakage, inference, and claim boundary | Frozen; qualification pending |
| `docs/research/hnph_v0.4_source_qualification_addendum.md` | Hash-bound fail-closed DOD admission contract | Frozen; qualification pending |
| `configs/protocol/hnph_phase0_v0.4.yaml` | Machine-readable active HNPH protocol | Frozen; claim mode false |
| `docs/audit/invalidated_result_registry.json` | INV-004 records the v0.3 supersession required by the B2 freeze rule | Active |
| `docs/figures/hnph_protocol/figure_manifest.json` | Figure classification, inputs, outputs, and captions | Protocol-only; no empirical result |

## Frozen Contract Ledger

| Contract | Canonical value | Implementation state | Evidence state |
| --- | --- | --- | --- |
| Development cohort | DOD-H after five-rater source qualification, person-grouped | Qualification tooling only | Unverified |
| External cohort | Sealed DOD-O after source qualification and model-family freeze | Qualification tooling only | Unopened |
| Descriptive transport | Sleep-EDF and CAP, single-label and non-claim-enabling | Not established | None |
| Endpoint | Oracle-conditional next stable Wake/NREM/REM transition | Not established by C0 | None |
| Cadence/context | 30-second natural grid; 10-minute causal context | Not established by C0 | None |
| Stability | Two consecutive destination epochs | Not established by C0 | None |
| Bands | `(0.5,2]`, `(2,5]`, `(5,10]`, `(10,20]` minutes | Not established by C0 | None |
| Primary endpoint | Subject-balanced incremental log skill in sealed DOD-O `(2,5]` minutes | Not established | None |
| Chief comparator | Validation-selected semi-Markov competing-risk model including current and two preceding macrostates | Not established by C0 | None |
| Test policy | One sealed external opening after qualification and complete freeze | Enforced by gate and figure refusals | DOD-O unopened |

## Invalidated Legacy Inputs

| Input class | Canonical disposition |
| --- | --- |
| Historical MSE 3.116 result | Overlapping-target GRU translator result; not future forecasting, NFC, SSM, or HNPH evidence |
| Failed seven-A100 execution | Invalid distributed run due to rank drift; no scientific result |
| Completed six-A100 execution | Invalid despite completion and reported metrics; gate false and data/evidence contract failures |
| Raw source caches used by invalid runs | Preserve read-only; do not delete or call invalid solely because derived outputs failed |
| Derived prepared data from invalid runs | Quarantine; never load through a canonical HNPH manifest |
| Checkpoints from invalid runs | Quarantine; no warm start, selection, comparison, or claim use |
| Evidence/results from invalid runs | Quarantine; metrics may appear only in the invalidation registry with `invalid` status |

## Downstream Gates

The following remain incomplete after C0 and block expensive training:

1. DOD-H/DOD-O five-rater, identity, license, physical-metadata, and source-hash qualification.
2. Canonical person identity map and zero-overlap split audit.
3. Physical unit, sampling-rate, channel, reference, and full-record length audits.
4. Rebuilt prepared records with no silent channel or sample truncation.
5. Frozen baseline budget and complete chief-comparator outputs.
6. Causal target, transition, calibration, negative-control, and subject scorer tests.
7. One-device, two-process lockstep, resume-parity, and rank-completeness tests.
8. Clean raw-to-evidence reproduction and independent score verification.
9. External-test seal and explicit red-team approval.
10. Multi-rater/soft-label reproducibility reference with the v0.3 frozen
    subject/rater support and max-t provenance.

Until every applicable gate is recorded as passing, `DOD-O opened=false`,
`training_authorized=false`, and `claim_eligible=false` remain controlling.

After B2, independent-replication wording remains blocked until a third cohort
and an unaffiliated falsification-oriented run complete the frozen replication
packet. That milestone does not open DOD-O or substitute for B2.
