# HNPH v0.2 Canonical Implementation Ledger

**Canonical protocol:** `kahlus.hnph.phase0.v0.2`
**Ledger date:** 2026-07-10
**Current authorization:** Documentation and data-governance remediation only

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

## Frozen Contract Ledger

| Contract | Canonical value | Implementation state | Evidence state |
| --- | --- | --- | --- |
| Development cohort | Sleep-EDF Expanded Sleep Cassette, person-grouped | Not established by C0 | None |
| External cohort | Sealed CAP Sleep Database | Not established by C0 | Unopened |
| Secondary shift cohort | Sleep-EDF Sleep Telemetry, descriptive and never pooled into training | Not established by C0 | None |
| Endpoint | Oracle-conditional next stable Wake/NREM/REM transition | Not established by C0 | None |
| Cadence/context | 30-second natural grid; 10-minute causal context | Not established by C0 | None |
| Stability | Two consecutive destination epochs | Not established by C0 | None |
| Bands | `(0.5,2]`, `(2,5]`, `(5,10]`, `(10,20]` minutes | Not established by C0 | None |
| Primary endpoint | Subject-balanced incremental log skill in CAP `(2,5]` minutes | Not established by C0 | None |
| Chief comparator | Validation-selected semi-Markov competing-risk model including current and two preceding macrostates | Not established by C0 | None |
| Test policy | One sealed external opening after complete freeze and red-team approval | Not established by C0 | CAP unopened |

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

1. Official-source file inventory and source SHA-256 manifest.
2. Canonical person identity map and zero-overlap split audit.
3. Physical unit, sampling-rate, channel, reference, and full-record length audits.
4. Rebuilt prepared records with no silent channel or sample truncation.
5. Frozen baseline budget and complete chief-comparator outputs.
6. Causal target, transition, calibration, negative-control, and subject scorer tests.
7. One-device, two-process lockstep, resume-parity, and rank-completeness tests.
8. Clean raw-to-evidence reproduction and independent score verification.
9. External-test seal and explicit red-team approval.

Until every applicable gate is recorded as passing, `CAP opened=false`,
`training_authorized=false`, and `claim_eligible=false` remain controlling.
