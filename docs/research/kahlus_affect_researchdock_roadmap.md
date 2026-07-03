# Kahlus-Affect / ResearchDock Roadmap

## Need Statement

A low-cost system is needed to quantify reward, effort, and stress-recovery response profiles using synchronized pupil, behavior, physiology, and optional neural signals so that anhedonia-like and trauma-related state changes can be explored more objectively than surveys alone.

This branch measures response profiles. It does not diagnose anhedonia, depression, PTSD, or any clinical condition.

## Existing At-Home Biosensors Are Ingredients, Not the Product

At-home and point-of-care saliva workflows already exist for cortisol and salivary alpha-amylase,
including self-collection, smartphone/strip/electrode readout, and comparison against lab assays.
Therefore, Kahlus RewardDock should not be framed as inventing a generic cortisol or stress meter.

The product wedge is disease- and task-specific interpretation:

- standardized reward/stress/social-response tasks
- pupil, behavior, HRV/PPG, and optional saliva biomarkers
- within-person trend modeling
- clinician-facing treatment-response summaries

## RewardDock Clinical Extension

Kahlus RewardDock Clinical =

- standardized task battery
- webcam pupillometry
- reaction time / effort behavior
- optional PPG/HRV
- optional cortisol / alpha-amylase module
- Kahlus response-profile model
- clinician dashboard

Primary clinical wedge:

Adults with anhedonia-related depression or social anxiety who are starting or adjusting treatment.

Need statement:

A way to objectively track reward-response and stress-recovery changes in adults undergoing
treatment for anhedonia-related depression or social anxiety in order to help clinicians identify
ineffective treatment plans earlier than self-report alone.

## Why This Is Not Just a Better Survey

The goal is not to replace interviews or self-report. The goal is to add repeated within-person
response data under standardized conditions.

Kahlus asks:

- Is this person's reward/stress/social-response profile improving relative to their own baseline?

Not:

- Does this person have depression?
- Does this person have anhedonia?

## Biomarker Architecture

Core v0:

- webcam pupil dilation
- reaction time
- effort persistence
- task accuracy
- self-report slider

Core v1:

- PPG/HRV
- heart-rate recovery
- optional EDA

Future biochemical add-on:

- salivary cortisol
- salivary alpha-amylase

Avoid as primary:

- dopamine
- oxytocin
- epinephrine/norepinephrine direct measurement

Dopamine, oxytocin, and epinephrine are not practical home biomarkers for this use case.
Peripheral levels do not cleanly map to central reward circuitry, assays are difficult or
context-dependent, and they are not condition-specific enough for RewardDock v0.

Cortisol and salivary alpha-amylase are more realistic stress-context add-ons, but they should not
be the core product.

## Clinical-First, Wellness-Later Product Path

Phase 1: Clinical/research use

- treatment-response tracking
- psychiatry clinics
- therapy programs
- clinical trials
- MDD/anhedonia/social anxiety cohorts

Phase 2: Research/remote monitoring

- longitudinal digital phenotyping
- stress-recovery studies
- reward learning studies

Phase 3: Consumer/self-improvement

- motivation tracking
- burnout recovery
- focus/reward habits
- social-confidence training

Do not start with self-improvement branding. Start clinical/research first to avoid becoming a
generic dopamine-maxxing wellness app.

## RewardDock Claim Boundaries

Allowed:

- tracks reward-response profile
- tracks stress-recovery profile
- tracks within-person change over time
- supports treatment-response research
- supports clinician review as an adjunct signal

Blocked:

- diagnoses depression
- diagnoses anhedonia
- diagnoses social anxiety
- recommends medication
- adjusts medication
- treats depression
- treats PTSD
- replaces clinician judgment
- claims cortisol/alpha-amylase alone measures anhedonia

## RewardDock Device Roadmap

RewardDock v0:

- webcam pupil + reward task + reaction time + self-report

RewardDock v1:

- add PPG/HRV

RewardDock v2:

- add optional saliva cortisol / alpha-amylase integration

RewardDock v3:

- clinician dashboard + longitudinal treatment-response report

RewardDock Consumer:

- future nonclinical self-improvement product only after clinical/research validation

## ResearchDock v0 Sensors

- webcam pupil/gaze
- reaction time
- task accuracy
- self-report sliders
- optional PPG/HRV
- optional EEG/fNIRS later

## Safe Task Battery

- reward anticipation
- probabilistic reward learning
- effort-for-reward
- mild frustration/stress task
- recovery/rest block
- visual attention block

No trauma exposure, treatment, stimulation, or unsupervised photic testing is part of RD-0.

## Metrics

- reward response delta
- reaction time change
- effort persistence score
- pupil response amplitude
- pupil recovery slope
- PPG/HRV proxy summary
- task accuracy summary
- response profile vector

## Baselines

- neutral-condition mean
- participant-level persistence
- ridge regression for response-profile prediction
- simple MLP after enough real data exists

## Evaluation Criteria

- no personally identifying fields
- synchronized task and sensor timestamps
- quality flags for missing pupil, high noise, dropped packets, and invalid reaction time
- deterministic synthetic fixtures
- finite metrics under missing sensors
- gate blocks clinical claims

## Buildable In 8 Weeks

- RD-0 synthetic schema, metrics, data card, and evidence gate
- local task app/session prototype with CSV/session export
- design-only webcam/pupil interface contract and quality flags
- optional PPG/HRV input schema
- public dataset mapping review for WESAD, DEAP, and SEED

## RD-1 Local Prototype Boundary

The first prototype is a local session protocol and export contract. It writes task/session CSVs and a hardware-free interface contract; it does not open a webcam, collect PPG, or ingest real participant data.

## RD-2 Synthetic Observation-Model Boundary

The RD-2 model lane uses synthetic ResearchDock sessions only. Behavior/task/self-report features predict pupil and HRV proxy observations under a subject-held-out split. Mean and ridge baselines are evaluated before the ResearchDock observation-operator candidate. This is synthetic pretraining infrastructure, not a clinical model.

## RD-3 Public Dataset Review Boundary

RD-3 maps WESAD, DEAP, and SEED to ResearchDock fields without adding loaders or downloading data. WESAD is the strongest immediate physiology/stress candidate; DEAP is useful for affective EEG plus physiology only after access terms are rechecked because the historical dataset page was unavailable during review; SEED is useful for EEG plus eye-movement missing-modality experiments after application and license approval.

The RD-3 artifacts are review outputs only. They do not contain raw participant data, do not claim diagnosis, and do not make any dataset ingestion path executable.

## RD-4 Pilot Preflight Boundary

RD-4 is a pre-collection readiness artifact for a future validation-scale ResearchDock pilot. It writes a pilot manifest, required-evidence checklist, and preflight gate using synthetic fixtures and the RD-1 protocol only.

RD-4 does not collect participant data, open hardware, provide legal or IRB advice, diagnose, treat, stimulate, or make clinical claims. Any real pilot remains blocked until appropriate review, local storage outside source control, identifier handling, and artifact archival are handled outside this code-only scaffold.

## RD-7 Observation Missing-Modality Report Counts

The synthetic observation-model report now includes per-reason missing-modality counts for skipped
trials. This keeps missing pupil, HRV proxy, sensor-packet, and behavior-response exclusions visible
in the human-readable artifact, not only in JSON metadata.

## RD-8 Observation Split Report Summary

The synthetic observation-model report now renders the subject-held-out split summary with train/test
subject counts and subject-overlap status. This keeps split hygiene visible beside baseline and
candidate metrics in the reviewer-facing artifact.

## RD-9 Observation Split-Audit Sidecar

The synthetic observation-model artifact writer now emits `researchdock_observation_split_audit.json`
with subject-held-out split counts, subject-overlap status, a leakage pass boolean, and explicit
failure reasons. This gives audit consumers a machine-readable split gate alongside the report.

## RD-10 Observation Report Split-Audit Verdict

The synthetic observation-model report now renders the split-audit verdict and any split-audit
failure reasons from the same helper that writes the JSON sidecar. This keeps blocked split evidence
reviewable without opening the sidecar first.

## RD-11 Aggregate Observation Artifact Index

The top-level ResearchDock synthetic report now indexes the RD-2 observation artifacts, including
`researchdock_observation_split_audit.json`. This makes the aggregate report point reviewers to the
machine-readable split audit, baseline table, metrics, and observation report.

## RD-12 ResearchDock Gate Criteria

The ResearchDock evidence gate now records branch-specific criteria in `researchdock_evidence_gate.json`
and renders them in the top-level report: allowed claim scope, blocked clinical/device terms, required
data-card pass, baseline table, finite metrics, calibration check, and synthetic-only boundary.

## RD-13 ResearchDock Failure-Reasons Sidecar

The top-level ResearchDock synthetic runner now writes `researchdock_failure_reasons.json` with gate
failures, data-card safety failures, and blocked claim terms. Passing synthetic fixtures still emit
the sidecar with empty failure lists so audit consumers have a stable artifact contract.

## RD-14 Top-Level Evidence Artifact Index

The top-level ResearchDock synthetic report now includes an `Evidence Artifact Index` listing the
metrics, data card, evidence gate, failure-reasons sidecar, and report artifacts. This makes the
local evidence bundle navigable from the Markdown report.

## RD-15 Top-Level Failure Reasons Summary

The top-level ResearchDock synthetic report now summarizes the same failure-reasons payload written
to `researchdock_failure_reasons.json`: gate failure count, data-card failure count, and blocked
claim-term count. This keeps passing and blocked bundles quick to audit from Markdown.

## RD-16 Top-Level Failure Reasons Details

The top-level ResearchDock synthetic report now renders failure-reason detail lists from the same
payload written to `researchdock_failure_reasons.json`, using `none` for empty gate/data-card failure
lists and listing blocked clinical/device claim terms explicitly.

## RD-17 Optional Artifact Index Expansion

The top-level ResearchDock synthetic report now expands its `Evidence Artifact Index` when optional
RD-1 session export, RD-2 observation model, RD-3 public dataset review, RD-4 pilot preflight, and
RD-5 profile-readiness artifacts are requested. This keeps optional local evidence bundles
navigable from the Markdown report instead of requiring stdout or directory inspection.

## RD-18 Top-Level Data Card Summary

The top-level ResearchDock synthetic report now renders bounded data-card fields directly in
Markdown: PII/real-data/clinical-label/stimulation booleans, modalities, synthetic profiles, and
quality flags such as missing pupil. This keeps safety and quality evidence visible without opening
`researchdock_data_card.json`.

## RD-19 Top-Level Quality Flag Counts

The top-level ResearchDock synthetic report now renders deterministic session-level quality flag
counts from the metrics table, including `missing_pupil` and `synthetic_high_noise`. Counts are
session-grained rather than trial-grained so the report matches the synthetic metrics table and does
not inflate repeated per-trial quality flags.

## RD-20 Aggregate RD-2 Baseline Ladder Summary

When the RD-2 synthetic observation model is requested, the top-level ResearchDock report now renders
the baseline ladder directly: mean baseline, ridge baseline, then ResearchDock observation operator,
with MSE/MAE plus the best-baseline verdict. This makes baseline results visible before the candidate
model in the aggregate evidence bundle.

## RD-21 Aggregate RD-2 Split Audit Summary

When the RD-2 synthetic observation model is requested, the top-level ResearchDock report now renders
the subject-held-out split audit summary: train/test subject counts, subject-overlap status,
leakage-pass verdict, and failure reasons. This keeps split hygiene visible in the aggregate bundle
without requiring reviewers to open the RD-2 sidecar first.

## RD-22 Aggregate RD-2 Missing-Modality Summary

When the RD-2 synthetic observation model is requested, the top-level ResearchDock report now renders
the missing-modality audit summary: total, eligible, and skipped trial counts plus per-reason skipped
trial counts. This keeps missing pupil and missing behavior-response exclusions visible at the
aggregate evidence-bundle entry point.

## RD-23 Aggregate RD-5 Readiness Audit Summary

When the RD-5 response-profile readiness audit is requested, the top-level ResearchDock report now
renders the readiness scope, clustering-disabled status, metric-row threshold, finite-vector verdict,
and failure reasons. This keeps future-clustering blockers visible without requiring reviewers to
open the RD-5 sidecar first.

## Long-Term

- real pilot usability study after review and consent workflow
- multimodal v2 observation model
- subject adaptation design
- ResearchDock to Kahlus v3 response-profile data flywheel

## Citations To Verify

| Reference | Why It Matters | citation_status |
| --- | --- | --- |
| Brendler et al. 2024 Scientific Reports pupil/reward/anhedonia | pupil reward-response precedent | needs_verification |
| Fietz et al. 2024 Biological Psychiatry CNNI pupil response profiles | pupil response-profile precedent | needs_verification |
| WESAD | stress physiology dataset candidate | verified_rd3_source_review |
| DEAP | affect physiology/EEG dataset candidate | partly_verified_rd3_dataset_page_unavailable |
| SEED | affect EEG and eye-tracking dataset candidate | verified_rd3_access_requires_application |
