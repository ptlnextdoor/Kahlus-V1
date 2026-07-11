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
| Primary success rule | The one-sided simultaneous 95% subject-bootstrap lower confidence bound for primary-band incremental log skill is greater than zero, external Brier score is noninferior to the chief comparator, and the frozen calibration, prediction-set, and negative-control gates pass |

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

The 0.02 bit/anchor value is a preregistered design-sensitivity target, not a
biological constant, minimum clinically important difference, or established
effect size. Failure to reach the power target stops model development under
this protocol rather than authorizing a larger architecture.

## Frozen Scoring and Inference

Let `K = 4` denote the three destination macrostates plus no event in the
reported band. The chief comparator uses Jeffreys smoothing (`alpha = 0.5` per
category) fitted only on development-training subjects. Before scoring, model
and comparator probabilities are evaluated in float64, clipped to
`epsilon = 1e-12`, and renormalized. For anchor `i` with observed category
`y_i`, incremental log skill is

`log2(p_model(y_i) / p_comparator(y_i))`.

Scores are averaged within person first and then equally across people. Windows
or anchors are never treated as independent people. The primary confidence
bound is a one-sided max-t 95% lower bound from a subject-cluster bootstrap over
the four frozen lead bands. Bootstrap resamples retain every anchor from a
selected person, preserving within-person dependence. The random seed, number
of replicates, and tie handling are frozen before the CAP test seal.

For discrete competing-risk training, `c` is the number of complete target bins
observed after the issue time. `S_0 = 1`; a record right-censored after `c`
complete bins contributes `-log(S_c)`. An event in bin `j` with destination `k`
contributes `-log(S_{j-1} h_{j,k})`. The Phase 0 primary score uses only anchors
with complete follow-up through the reported band; censoring-aware likelihood is
a training contract and a prespecified sensitivity analysis, not a way to label
censored anchors as no-event outcomes.

The external calibration gate requires all of the following:

- the subject-bootstrap 95% interval for multiclass calibration intercept
  contains `0` and the interval for calibration slope contains `1`;
- the one-sided 95% upper bound for top-label expected calibration error is at
  most `0.10`;
- a validation-fitted 90% adaptive prediction set has an external
  subject-bootstrap 95% coverage interval containing `0.90`, median set size
  below `K`, and full-set frequency below `0.50`.

External Brier noninferiority is defined as a one-sided 95% lower confidence
bound of `Brier_comparator - Brier_model >= 0`. Each negative control must have
a one-sided 95% upper confidence bound on incremental log skill no greater than
zero. Calibration thresholds are protocol constants for falsification; they do
not establish clinical utility.

The transition frontier is empty when the first frozen band fails its
simultaneous lower-bound or calibration gate. Otherwise it ends at the last
band in the longest contiguous prefix that passes. A favorable isolated later
band cannot skip a failed earlier band.

## Freeze and Test Seal

Before CAP is opened, the following must be frozen and hashed: official source
file manifest, person grouping, ontology and annotation mapping, context and
anchor contract, lead bands, chief comparator family and tuning budget, model
family and tuning budget, validation-only calibration method, report template,
software commit, and environment. CAP labels or metrics may not select features,
hyperparameters, checkpoints, calibration, exclusions, or claims.

CAP is an external-dataset test, not independent replication. Reversing the
development and external cohorts is a sensitivity analysis. Any claim of
independent replication requires a third cohort whose preprocessing, subjects,
recordings, and test decisions were not used for Sleep-EDF or CAP development.

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
  prediction-set, Brier, provenance, leakage, comparator, control, distributed,
  and reproduction gates.
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

If the chief semi-Markov comparator ties or outperforms the neural model on the
internal or external primary endpoint, architecture expansion stops. The result
is reported as a bounded or calibrated-null finding rather than used to justify
additional model capacity.

## Historical Invalidation Boundary

The historical MSE 3.116 result, the failed seven-A100 run, and the completed but
invalid six-A100 run are not HNPH v0.2 evidence. Their exact dispositions are in
`docs/audit/INVALIDATED_RESULT_REGISTRY.md` and
`docs/audit/invalidated_result_registry.json`. Raw caches are retained; affected
prepared datasets, checkpoints, and evidence outputs are quarantined from all
canonical selection, comparison, and reporting paths.
