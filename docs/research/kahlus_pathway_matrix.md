# Kahlus Pathway Matrix

| Pathway | Need Statement | Build Now | Long-Term | Blocked Claims |
| --- | --- | --- | --- | --- |
| Kahlus v1 | Make EEG forecasting/evaluation defensible under leakage-proof splits and strong baselines. | baseline ladder, subject-held-out audits, autocorrelation controls | public EEG benchmark expansion | SOTA, diagnosis, treatment |
| Kahlus v2 | Model shared latent brain/body state through observation operators. | synthetic multimodal fixtures and operator tests | real multimodal observation model | clinical digital twin, foundation model proven |
| Kahlus v3 | Model perturbation-response profiles under structured tasks. | Transition Gym and synthetic operator recovery | ResearchDock data flywheel | recovery/diagnosis claims without baseline win |
| ResearchDock / Kahlus-Affect / RewardDock | Track reward-response and stress-recovery changes under standardized tasks for treatment-response research in anhedonia-related depression or social anxiety. | RD-0 synthetic schemas, metrics, data card, gate; RewardDock v0 webcam pupil + reward task + reaction time + self-report | PPG/HRV, optional cortisol/alpha-amylase add-on, clinician dashboard, longitudinal treatment-response reports, later nonclinical self-improvement | depression/anhedonia/social-anxiety diagnosis, medication recommendations, treatment claims, cortisol/alpha-amylase-alone anhedonia claims |
| Kahlus-NeuroVisual | Structure visual-perceptual episodes without diagnosis. | roadmap and public dataset review | supervised research annotation | epilepsy diagnosis, self-triggered photic testing |
| Kahlus-EM | Separate true neural-state changes from artifacts and environmental confounds. | artifact audit only | confound-control reporting | EM stimulation, consciousness claims |
| Kahlus-Sleep | Future dream cueing / memory and affect recovery branch. | do not implement now | protocol research after safety review | treatment claims |
| Orch-OR / quantum biology | Speculative appendix only. | docs appendix only | none in code | quantum consciousness device |

## Cross-Cutting Evaluation Criteria

- leakage audit before model claims
- baselines before main models
- finite metrics
- quality flags
- dataset provenance
- explicit claim boundary
- no raw private participant data committed

## Citation Verification Policy

References that are not already verified in local repo context must be marked `citation_status: needs_verification`. RD-0 does not require web calls or citation expansion.

## RewardDock Product Boundary

At-home and point-of-care saliva workflows already exist for cortisol and salivary alpha-amylase,
including self-collection, smartphone/strip/electrode readout, and comparison against lab assays.
Kahlus RewardDock should treat those biosensors as ingredients, not as the product.

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

Kahlus asks whether a person's reward/stress/social-response profile is improving relative to their
own baseline. It does not ask whether the person has depression, anhedonia, or social anxiety, and it
does not replace interviews, self-report, or clinician judgment.

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

Dopamine, oxytocin, and epinephrine are not practical home biomarkers for this use case. Peripheral
levels do not cleanly map to central reward circuitry, assays are difficult or context-dependent, and
they are not condition-specific enough for RewardDock v0. Cortisol and salivary alpha-amylase are
more realistic stress-context add-ons, but they should not be the core product.

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

## Updated RewardDock Claim Boundaries

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
