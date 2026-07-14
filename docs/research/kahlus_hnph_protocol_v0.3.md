# Kahlus HNPH Phase 0 Protocol v0.3

**Status:** frozen preregistration before claim-mode evaluation
**Protocol ID:** `kahlus.hnph.phase0.v0.3`
**Effective date:** 2026-07-13
**External test opened:** no

This protocol supersedes v0.2 only for the B2 preregistration additions below.
The v0.2 C0 freeze remains preserved as the historical decision record. No
held-subject or CAP claim-mode outcome is included in this version.

The machine-readable contract is
`configs/protocol/hnph_phase0_v0.3.yaml`; the controlling detail for these
additions is the [B2 preregistration addendum](hnph_b2_preregistration_addendum.md).
The full Phase 0 rationale remains in
[the v0.2 protocol](kahlus_hnph_protocol_v0.2.md) and
`kahlus_hnph_full_development_protocol.md`, except where this version changes a
frozen B2 rule.

## Frozen B2 decision boundary

`epsilon = 0.02` bits per anchor is a preregistered design-sensitivity floor
for subject-balanced out-of-sample log-skill. It is neither a clinical effect
size nor a biological threshold.

An internal B2 pass authorizes only implementation of a fixed, small H3 model
family. It does not open CAP, authorize a neural claim, or permit external data
to select the model. Before CAP claim-mode evaluation, the H3 family,
calibration, report template, scoring code, and split contract must be frozen.

Any positive external claim requires every named external dataset to meet the
frozen integrity, power, comparator, control, construct-validity, and
residual-skill gates on at least one preregistered band with complete follow-up.
A gate failure is a bounded/null result and stops architecture expansion.

Power is calculated at the subject-cluster level with `d = 0.02`, familywise
`alpha = 0.05`, target power `0.80`, and a per-subject standard deviation frozen
from training folds or synthetic data before claim-mode evaluation. An
underpowered band may report its bounded interval but cannot authorize H3.
The frozen internal B2 family has 12 hypotheses: four lead bands by three
transition types by one named internal evaluation.

## Primary operational target

The endpoint remains the next stable transition, not a separately defined
future-state classifier. Its v0.3 primary outcome is five-way:

```text
{no_event, Wake, NREM, REM, Ambiguous}
```

`Ambiguous` is a scored outcome for observed annotation ambiguity, an `Unknown`
interval, or an unresolved non-current boundary. A one-epoch excursion that
returns to the current macrostate under the fixed two-epoch stability rule is
`no_event`. Right-censored anchors without complete follow-up are excluded only
from that band and their count is reported. Neither difficult labels nor
censoring may be silently deleted.

This target is an operational sleep-scoring endpoint. It is not a claim that a
single biological transition occurs at one precisely observed instant.

## Chief comparator

The cause-specific semi-Markov competing-risk nuisance comparator is a blocking
deliverable. It must pass training-only dwell-time fit, held-subject calibration
and Brier reporting, a no-EEG synthetic semi-Markov test, and the simpler
persistence/Markov ladder.

It must also pass a frozen nuisance-challenger adequacy audit: no held-subject
nuisance-only challenger may show a subject-cluster lower-bound advantage of
`epsilon` or more over the chief comparator. This is a finite falsification
test, not an asserted empirical upper bound on real-data `E KL(p0||q0)`.

## Label construct-validity gate

Before H3 authorization, each claim band requires a
**label-reproducibility reference** using at least three independent raters or
validated leave-one-rater-out soft labels/hypnodensity with equivalent
provenance. The held-out rater cannot contribute to their own target. Reference
log-skill is scored against the same frozen per-anchor chief nuisance comparator
as B2 and uses the same subject-cluster max-t inference.

The reference is not a mathematical ceiling: a model can learn one scorer's
convention more reliably than scorers agree with each other. It is a gate on the
construct being measured. Missing provenance or a reference lower bound below
`epsilon` blocks H3. Literature-average agreement, a single global kappa, or
auto-scorer confidence is insufficient.

## Evidence and replication

Every B2 run emits hash-bound JSON and Markdown artifacts. In addition to the
standard integrity evidence, they record the addendum hash, `epsilon`, frozen
power inputs, comparator acceptance and nuisance-challenger lower bounds,
ambiguity/censoring accounting, and label-reference provenance. Invalid or
incomplete evidence exits nonzero; a complete bounded/null result exits zero.
The gate opens the local typed artifacts and verifies their hashes and decision
fields; a caller-provided hash string or pass flag is not evidence.

After B2, an independent-replication claim requires a versioned replication
packet and an unaffiliated group that is invited to attempt to falsify any
positive finding. CAP is external validation, not independent replication;
reversing Sleep-EDF and CAP is sensitivity analysis only. Independent
replication additionally requires a third cohort untouched by Sleep-EDF/CAP
development and the same frozen gates.

## Claim boundary

This protocol supports only a preregistered, leakage-aware benchmark and its
bounded or gated results. It does not support clinical, diagnostic, treatment,
seizure-warning, digital-twin, foundation-model, biological-mechanism, or
consciousness claims.
