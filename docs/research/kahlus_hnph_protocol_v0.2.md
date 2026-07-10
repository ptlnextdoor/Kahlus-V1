# Kahlus HNPH Phase 0 Protocol v0.2

**Status:** Canonical protocol freeze; implementation and results are not implied
**Protocol ID:** `kahlus.hnph.phase0.v0.2`
**Effective date:** 2026-07-10
**External test opened:** no

This document is the concise, controlling Phase 0 contract. The full scientific
rationale remains in `kahlus_hnph_full_development_protocol.md`. If an older
configuration, prepared dataset, checkpoint, result, or report conflicts with
this freeze, the older artifact is ineligible for HNPH v0.2 evidence.

## Frozen Scientific Question

Does causal EEG history add externally generalizable information about the next
stable Wake/NREM/REM transition beyond an oracle-conditional semi-Markov
competing-risk comparator?

"Oracle-conditional" means that the evaluator and chief comparator may condition
on the true current scored macrostate and its causally available history. This is
a controlled scientific endpoint, not a deployable label-free forecasting claim.

## Frozen Phase 0 Contract

| Item | Frozen value |
| --- | --- |
| Development data | Sleep-EDF Expanded Sleep Cassette records only |
| Development use | Train, validation, held-subject internal test, target and calibration development, model and baseline selection |
| External data | CAP Sleep Database, sealed and untouched during development |
| Secondary shift data | Sleep-EDF Sleep Telemetry; descriptive only and never pooled into training |
| Split unit | Person; all nights from one person stay in one split |
| Macrostate ontology | Wake, NREM, REM, Unknown; R&K 1-4 and AASM N1-N3 map to NREM |
| Evaluation cadence | Natural 30-second epoch grid; no event-enriched evaluation |
| Causal EEG context | 10 minutes ending at the issue time |
| Stable transition | Destination differs from current macrostate and persists for two consecutive 30-second epochs |
| Transition lead bands | `(0.5,2]`, `(2,5]`, `(5,10]`, `(10,20]` minutes |
| Primary band | `(2,5]` minutes |
| Chief comparator | Validation-selected semi-Markov competing-risk model using current macrostate, two preceding macrostates, current bout age, elapsed recording time/time of night, recent transition count, and empirical destination/base-rate information |
| Primary estimand | Mean subject-balanced incremental categorical log skill in bits in the 2-5 minute band on sealed CAP, model versus chief comparator |
| Primary success rule | The subject-level 95% lower confidence bound for primary-band incremental log skill is greater than zero and the frozen calibration gate passes |

For every band, the categorical outcome includes Wake, NREM, REM destinations
and no event in band. Complete-follow-up natural-grid anchors are the Phase 0
default. Earlier, later, and absent transitions remain in the denominator as the
frozen evaluator specifies.

The transition frontier is secondary: it is the longest contiguous prefix of
the frozen bands for which subject-level lower confidence bounds are positive
and calibration gates pass. The primary endpoint remains the 2-5 minute CAP
endpoint even if another band is more favorable.

Baseline feasibility authorizes model work only if subject-clustered simulation
estimates at least 80% power for a 0.02 bit/anchor primary-endpoint gain. SSF-SET
is required related work and, when reproducible under this endpoint, a protocol
comparator; it does not establish a Kahlus first-sleep-forecaster claim.

## Freeze and Test Seal

Before CAP is opened, the following must be frozen and hashed: official source
file manifest, person grouping, ontology and annotation mapping, context and
anchor contract, lead bands, chief comparator family and tuning budget, model
family and tuning budget, validation-only calibration method, report template,
software commit, and environment. CAP labels or metrics may not select features,
hyperparameters, checkpoints, calibration, exclusions, or claims.

All transforms are split after person assignment and fit on training data or
causal context as applicable. Input tensors must exclude subject IDs, dataset
IDs, paths, dates, source keys, future labels, and future samples. Physical
units, sampling rates, channel identities, references, masks, and transform
lineage must remain explicit; silent harmonization or truncation is forbidden.

## Primary Claims and Outcomes

Before a valid sealed evaluation, the only allowed statement is that HNPH Phase
0 is a preregistered, leakage-audited evaluation protocol. No positive HNPH or
Sleep Transition Frontier result has been established.

After evaluation, claim wording is controlled by the machine-readable gate:

- **Full:** positive primary-band external skill, passing calibration and all
  provenance, leakage, comparator, control, distributed, and reproduction gates.
- **Limited:** a prespecified supporting endpoint passes while the primary
  endpoint does not; report only that endpoint and its limitation.
- **Calibrated null:** the evaluator is valid but the model does not beat the
  chief comparator; report the limit, not model superiority.
- **Invalid:** any split leakage, causal violation, provenance failure, missing
  chief comparator, failed calibration/control, rank drift, or irreproducible
  evidence invalidates the run and its metrics for claims.

This protocol does not authorize clinical, diagnostic, treatment, seizure
warning, digital-twin, foundation-model, biological-mechanism, or site-held-out
claims. CAP is a dataset-held-out external evaluation unless independent site
provenance supports stronger wording.

## Historical Invalidation Boundary

The historical MSE 3.116 result, the failed seven-A100 run, and the completed but
invalid six-A100 run are not HNPH v0.2 evidence. Their exact dispositions are in
`docs/audit/INVALIDATED_RESULT_REGISTRY.md` and
`docs/audit/invalidated_result_registry.json`. Raw caches are retained; affected
prepared datasets, checkpoints, and evidence outputs are quarantined from all
canonical selection, comparison, and reporting paths.
