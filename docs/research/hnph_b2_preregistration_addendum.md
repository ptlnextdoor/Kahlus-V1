# HNPH B2 Preregistration Addendum

**Status:** frozen before the first held-subject or external claim-mode outcome.
**Frozen v0.3 protocol SHA-256:** `89dd1df781b089fe3fb25e8091ac7afc87817e8b466881170f665519cb3bbd08`
**Applies to:** HNPH B2 baseline feasibility, its evidence packet, and the
superseding `kahlus.hnph.phase0.v0.3` protocol. It leaves the merged v0.2
freeze intact. A later change requires a new addendum version and an
invalidation-registry entry.

## 1. Frozen detection floor and authority boundary

`epsilon = 0.02` bits per anchor is a preregistered design-sensitivity floor for
subject-balanced, out-of-sample log-skill. It is not a clinical threshold, a
biological constant, an interpretation of conditional mutual information, or a
consequence of the probability-clipping floor.

The score is the classical EEG-plus-nuisance model minus the validated nuisance
comparator, averaged within subject and then equally across subjects. The bound
is a one-sided 95% simultaneous subject-cluster max-t bound with 2,000
resamples.

There are two deliberately separate decisions:

- A development B2 result may authorize implementation of a fixed, small H3
  model family only when the internal integrity, power, comparator, control,
  construct-validity, and residual-skill gates pass.
- It never opens or tunes on CAP. External claim-mode evaluation is allowed only
  after the H3 family, calibration, report template, and scoring code are frozen.
  Requiring a CAP outcome before choosing H3 would turn the sealed test into a
  development signal.

At external claim mode, a positive result additionally requires every named
external dataset to have at least one preregistered lead band with complete
follow-up whose lower bound meets or exceeds `epsilon`; otherwise the outcome
is a bounded/null result and architecture expansion stops.

Subject-cluster power uses `d = epsilon`, familywise `alpha = 0.05`, target
power `0.80`, and a per-subject score-difference standard deviation estimated
only from training folds or synthetic data before claim-mode evaluation. If the
required number of event subjects exceeds the available number, the band is
declared underpowered before evaluation and can only produce a bounded interval.
For internal B2, the frozen family count is `|G| = 12`: four lead bands by three
transition types by one named internal evaluation.

## 2. Chief comparator acceptance

The chief null is a cause-specific semi-Markov competing-risk model, conditioned
only on current macrostate, dwell time, causal transition history, clock-time,
and declared homeostatic proxies. "Circadian" must not be claimed unless a
validated phase model is separately provided.

It is a blocking deliverable. Before it can serve as the chief comparator, the
artifact must demonstrate all of the following:

1. Training-only dwell-time fit against empirical survival by state.
2. Held-subject calibration and reported Brier decomposition.
3. A no-EEG synthetic semi-Markov world in which the EEG model has no residual
   gain.
4. Held-subject performance at least as strong as persistence and first/second
   order Markov baselines.
5. A preregistered nuisance-challenger adequacy audit: no held-subject nuisance
   challenger improves the chief comparator by `epsilon` or more in a claim
   band.

This is not an estimable upper bound on the real-data `E KL(p0||q0)` term and
must never be presented as one. It is a finite adequacy audit: a failed or
unavailable audit blocks H3 authorization.

## 3. Primary target and censoring

The frozen v0.2 endpoint is the next stable transition, so it retains its
explicit no-event outcome. Its primary alphabet is therefore five-way:

```text
{no_event, Wake, NREM, REM, Ambiguous}
```

Replacing `no_event` with the current macrostate would create a different
future-state estimand and require a new protocol version, comparator, and
evaluator. It is not a clarifying addendum to this transition endpoint.

`Ambiguous` covers an observed annotation ambiguity (including an `Unknown`
interval) or an unresolved non-current boundary that prevents a stable future
macrostate from being adjudicated. A one-epoch excursion that returns to the
current macrostate under the fixed two-epoch stability rule is `no_event`, not
`Ambiguous`. Every model must assign `Ambiguous` probability. Hard epochs are
never silently removed from the denominator. An anchor is excluded from a lead
band only when complete follow-up through that band is unavailable; its
right-censor exclusion is counted and reported by band.

Sensitivity analyses report a prespecified missingness treatment and 30-second
and 90-second stability definitions. A positive result requires agreement in
sign with these sensitivity analyses.

## 4. Required evidence fields

The HNPH evidence packet records the addendum hash, `epsilon`, its
design-sensitivity effect-size role, frozen power inputs, negative-control
upper and real-minus-control lower confidence bands, synthetic-known-signal and
nuisance-probe checks, comparator acceptance checks, the held-subject
nuisance-challenger lower bound by band, primary ambiguity handling,
complete-follow-up exclusions by band, label-reference provenance, claim scope,
and one of the frozen stop reasons:

```text
underpowered | integrity_fail | baseline_fail | control_fail |
residual_below_epsilon | comparator_challenger_exceeds_epsilon |
label_reliability_unavailable | label_reliability_below_epsilon |
pass_authorize_h3
```

Invalid or incomplete evidence exits nonzero. A complete bounded/null result
exits zero and remains a reportable result.

The gate opens the local runner artifacts named by the packet, recomputes their
SHA-256 values, and checks their typed fields against the decision inputs.
Hash-shaped strings or caller-supplied Booleans alone cannot authorize H3. The
canonical v0.3 protocol path and the protocol SHA-256 recorded at the top of
this addendum are both required.

## 5. Label construct-validity gate

Manual hypnograms are operational labels, not direct observations of a unique
instantaneous neural state. Therefore HNPH must measure a **label-
reproducibility reference** before using the target to authorize H3.

The reference is not a universal mathematical ceiling: a model may learn one
scorer's systematic convention better than independent scorers agree. Instead,
it measures how much of the reported forecastability is plausibly tied to the
operational label. Positive HNPH wording remains limited to the operational
target unless an independent construct-validity study supports stronger claims.

For each claim band, use at least three repeated independent scorer labels or a
held-out-rater soft-label/hypnodensity estimate with equivalent provenance.
Predict one rater's label from the remaining raters, calculate subject-balanced
log-skill against the **same frozen per-anchor chief nuisance comparator** used
by B2, and form the same subject-cluster max-t lower bound. The primary soft
target must be leave-one-rater-out; a target rater may not contribute to their
own consensus probability. Both the primary scorer and the label-reference
artifact bind the same soft-target provenance hash. Scores use the primary
five-way alphabet and the same float64 probability floor (`1e-12`) as the B2
log score.

H3 is blocked when multi-rater or independently validated soft-label evidence
is unavailable, or when the lower reference bound is below `epsilon`. Report
the classical model's skill as a fraction of the reference only when the
reference is positive and finite. A single global kappa, a literature average,
or an auto-scorer's confidence alone is not an acceptable substitute.

This is a construct-validity gate, not a claim that hypnodensity recovers a
single biological transition instant.

## 6. Independent replication milestone

After B2 is complete, before any "independently replicated" wording, HNPH
must publish a versioned replication packet and obtain a run by at least one
unaffiliated group invited to try to falsify a positive result. The packet must
bind the protocol version, evaluator and gate code, permitted source-manifest
instructions, preprocessing contract, and evidence verifier; raw neural data
remain outside git.

CAP is an external-dataset validation, not independent replication. Reversing
Sleep-EDF and CAP is a sensitivity analysis only. An independent replication
claim additionally requires a third cohort not used for Sleep-EDF or CAP
development and the unaffiliated run to report the same frozen gates, including
the label-reproducibility reference. This milestone does not open CAP or block
the internal B2 decision to implement a frozen H3 family; it blocks only
independent-replication wording until completed.
