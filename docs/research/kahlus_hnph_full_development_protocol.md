# Kahlus Human Neural Predictability Horizon

## Full Research and Development Protocol

**Protocol status:** Proposed governing specification
**Protocol version:** 0.1.0
**Date:** 2026-07-09
**Repository:** `Kahlus-V1` / Python package `neurotwin`
**Primary program name:** Kahlus Human Neural Predictability Horizon (HNPH)
**First biological endpoint:** Kahlus Sleep Transition Frontier (KSTF)
**Current evidence state:** Protocol design only. No HNPH or KSTF result has been established.
**Clinical state:** Non-clinical research. No diagnosis, warning, treatment, or clinical-decision claim is permitted.

---

## 1. Purpose of This Document

This document is the governing research and development protocol for the next Kahlus program. It is intended to be usable by a principal investigator, neuroscience collaborator, statistician, machine-learning engineer, data engineer, cluster operator, red-team reviewer, and publication lead without relying on undocumented conversation history.

It defines:

- the scientific problem and need;
- the pitch and claim boundary;
- the biological and neurological rationale;
- the measurement physics of scalp EEG;
- the mathematical targets, losses, scores, and statistical tests;
- the computer-science and machine-learning architecture;
- the public-data and prospective-data strategy;
- the baseline ladder and adversarial controls;
- the complete sprint sequence from protocol freeze to external evidence;
- local, single-GPU, and A100 progression gates;
- anticipated bugs, blockers, failure modes, and mitigations;
- evidence-bundle, publication, and reproducibility requirements;
- long-term multimodal, wearable, invasive-validation, and perturbation paths;
- terminal success, limited-result, null-result, and invalid-run outcomes.

This protocol does **not** guarantee a positive model result. It guarantees that the program ends each stage with an auditable scientific result or a documented kill decision. A failed model is allowed. An invalid or retrospectively redefined experiment is not.

### 1.1 Document Map

| Section range | Purpose |
| --- | --- |
| 2-7 | Program decision, pitch, need, claims, and falsifiable hypotheses |
| 8-10 | Biology, measurement physics, signal processing, and mathematical contract |
| 11-13 | Baselines, datasets, splits, leakage control, and external-test sealing |
| 14-16 | Proposed software interfaces, model architecture, and statistical analysis |
| 17-20 | Development sprints, compute escalation, verification, and evidence artifacts |
| 21-25 | Known bugs, blockers, risks, governance, ethics, privacy, and regulation |
| 26-27 | Wearable/multimodal extensions and publication strategy |
| 28-32 | Acceptance criteria, completion conditions, immediate actions, references, and governing rule |

Terms such as **current**, **implemented**, **proposed**, **frozen**, and **validated** are not interchangeable in this document. A proposed interface is not implemented code; an implemented component is not scientifically validated; and a positive internal result is not an external result.

---

## 2. Executive Decision

Kahlus will no longer be organized around the claim that a particular architecture called the Neural Field Compiler is inherently novel or biologically meaningful. The architecture must become a competitor inside a frozen evaluator.

The program will instead answer one central question:

> **How far into the future does noninvasive human neurophysiology contain calibrated, generalizable predictive information beyond strong causal baselines, and at what lead time does that information become sufficient to forecast a stable brain-state transition in people and acquisition systems not seen during training?**

The overarching scientific object is the **Human Neural Predictability Horizon**. The first biologically grounded endpoint is the **Sleep Transition Frontier**, defined as the externally validated, calibrated lead-time frontier for the next stable transition among Wake, NREM, and REM.

The program separates three regimes that must never be collapsed into one favorable number:

| Regime | Default lead range | Target | Scientific role |
| --- | ---: | --- | --- |
| Waveform-scale | 0.25-2 s | Strictly future waveform or complex spectral field | Sanity check for continuity, causality, and implementation |
| State-scale | 4-32 s | Future multiscale neurophysiological state summaries | Supporting predictive-dynamics claim |
| Transition-scale | 0.5-20 min | Time and destination of next stable sleep macrostate transition | Primary Phase 0 biological endpoint |

Only the state-scale and transition-scale regimes can support the scientific program. Short-horizon waveform results are necessary diagnostics but cannot establish a useful neural world model.

---

## 3. Thirty-Second Pitch

Most EEG machine-learning systems classify the present or reconstruct nearby samples. These tasks can look impressive while exploiting signal continuity, overlapping windows, subject identity, acquisition hardware, or poorly calibrated uncertainty. Kahlus will build an open evidence standard that asks a harder question: how early can a model forecast a genuinely future neurophysiological state or transition in a person and recording system it has never seen, beyond linear dynamics and state-duration priors, while knowing when it is uncertain? The first test will use dense, repeated sleep transitions because they support a powered public-data experiment without making a clinical claim. The long-term outcome is a validated predictability frontier and open evaluation system for trustworthy predictive neurotechnology.

---

## 4. Principal-Investigator-Level Need Statement

### 4.1 Formal Need Statement

Researchers and neurotechnology developers lack a validated, acquisition-aware method for determining how much generalizable information about future human brain-state dynamics is present in noninvasive EEG and how far in advance that information becomes predictive. Existing EEG forecasting and representation-learning studies often evaluate adjacent or overlapping windows, optimize point-error metrics dominated by short-range autocorrelation, and report within-dataset performance without subject-level uncertainty calibration or truly external validation. Consequently, it remains unclear whether reported gains reflect neural dynamics, state-duration and circadian priors, acquisition identity, or leakage.

There is a need for an openly specified, leakage-audited, externally validated framework that quantifies the calibrated lead-time frontier for future noninvasive neurophysiology beyond strong linear dynamical, semi-Markov, and modern neural baselines. Such a framework must separate waveform continuation from state forecasting, preserve physical time and electrode geometry, treat missingness and artifacts explicitly, perform inference at the subject level, and fail closed when calibration or negative controls do not pass. Establishing this frontier is a prerequisite for scientifically justified patient-specific, multimodal, wearable, or closed-loop studies.

### 4.2 One-Sentence Need Statement

> **A way to determine, under externally held-out conditions, how far in advance noninvasive neurophysiology contains calibrated information about an impending brain-state transition beyond autocorrelation, state duration, circadian timing, and acquisition identity.**

### 4.3 Human Need

People and clinicians experience neurological and physiological transitions as events that occur over time, but most available systems either identify an event after it begins or produce retrospective risk scores whose calibration and external validity are uncertain. A trustworthy forecast must answer not only "what may happen" but also "how early," "with what uncertainty," "for whom," "under which acquisition conditions," and "when should the model abstain." Kahlus addresses the measurement and evidence problem that must be solved before any patient-facing forecast can be responsibly studied.

---

## 5. Why This Problem Matters

### 5.1 Scientific Importance

The recorded future provides an objective answer key, but only if the task is constructed causally. Quantifying the forecastability frontier would separate:

- predictable neural or physiological structure;
- trivial waveform persistence;
- state-duration and time-of-night priors;
- acquisition and montage fingerprints;
- subject identity;
- artifact-driven changes;
- model miscalibration;
- genuinely unpredictable or weakly observed dynamics.

The output is not merely a better model. It is a measured boundary on what a declared class of noninvasive recordings and models can support.

### 5.2 Engineering Importance

A frozen evaluator can guide:

- which sensors and montages preserve predictive information;
- what history length is useful;
- whether more compute improves generalization or only memorization;
- whether a low-density wearable retains the state-scale frontier;
- when uncertainty becomes too broad for a forecast to be useful;
- whether a proposed transition or intervention study is justified.

### 5.3 Global Impact Path

EEG and polysomnography are more accessible than implanted electrodes, MEG, or high-field functional imaging. If the frontier is reproducible, an open evaluator and compact reference model could lower the barrier for laboratories and hospitals to test predictive neurophysiology under their own hardware and populations. This is an infrastructure and evidence impact before it is a device impact.

The program must not claim that a retrospective public-data result will improve health outcomes. Global impact requires prospective, multi-site evidence and, for any clinical use, a separate regulated validation program.

---

## 6. Claim Hierarchy

### 6.1 Claims Allowed After Protocol Implementation but Before Positive Evidence

- Kahlus implements a leakage-audited forecastability protocol.
- Kahlus evaluates waveform-, state-, and transition-scale prediction separately.
- Kahlus contains public-data adapters, baseline runners, negative controls, and evidence gates.
- Kahlus produces subject-level calibration and generalization reports.

### 6.2 Claims Requiring a Phase 0 Pass

- Kahlus demonstrates positive external forecast skill for the specified target and lead band.
- Kahlus improves on the strongest frozen comparator under the specified data contract.
- The improvement remains calibrated and passes the declared negative controls.
- The result generalizes to the declared external dataset without test-time retuning.

### 6.3 Claims Requiring Prospective Multi-Site Evidence

- The measured frontier replicates under newly collected data.
- Calibration survives different operators, hardware, populations, and missing-channel patterns.
- Forecasts can be emitted prospectively at the declared issue times.

### 6.4 Claims Explicitly Blocked

- diagnoses epilepsy, sleep disorders, depression, anhedonia, or any disease;
- predicts or warns of an individual's seizure;
- recommends or adjusts treatment;
- supports autonomous clinical decisions;
- reconstructs implant-equivalent brain activity;
- discovers a biologically identified latent brain state;
- models consciousness;
- is a validated digital twin or brain foundation model;
- proves causal effects from observational forecasting;
- treats or prevents a neurological event;
- replaces clinician judgment or polysomnographic scoring.

---

## 7. Scientific Program and Falsifiable Hypotheses

### 7.1 Primary Program Hypothesis

For continuous noninvasive human EEG, a causal, geometry-aware probabilistic model can extract incremental information about future state and stable sleep transitions that is not captured by the strongest validation-selected causal baseline, and this incremental information remains positive and calibrated on unseen subjects from an unseen acquisition dataset.

### 7.2 Null Hypothesis

After controlling for persistence, linear autoregression, state duration, circadian/time-of-recording structure, signal quality, subject identity, acquisition identity, and input-target overlap, no externally generalizable incremental information remains.

### 7.3 Supporting Hypotheses

| ID | Hypothesis | Required evidence | Failure meaning |
| --- | --- | --- | --- |
| H-WAVE | Strictly future waveform forecasts beat persistence and linear dynamics at short leads | Subject-level skill curve with no overlap | Implementation and continuity sanity only |
| H-STATE | State-summary forecasts retain positive skill at two consecutive leads at or beyond 4 s | External energy/log-score skill and calibration | No general state-scale claim |
| H-TRANS | EEG history adds information about the next stable sleep transition beyond an oracle semi-Markov baseline | External 2-5 min log-skill gain | No neural transition-frontier claim |
| H-GEN | Skill survives an untouched external dataset and reciprocal direction | Sealed test and reverse replication | Dataset-specific result only |
| H-CAL | Predictive probabilities remain calibrated near and away from transitions | Conditional calibration and Brier/log scores | No usable probability claim |
| H-GEO | Geometry and missingness handling improve robustness without encoding dataset identity | Ablations and identity probes | Simplify model or narrow hardware claim |
| H-PROS | Frontier replicates prospectively at independent labs | Preregistered prospective score packets | Retrospective-only finding |

### 7.4 Outcome Classes

Every completed study must end in exactly one class:

1. **Full pass:** state-scale and transition-scale primary gates pass externally.
2. **Dynamics-only pass:** state-scale gate passes, transition gate fails. Transition claims are killed.
3. **Transition-prior result:** semi-Markov/time-of-night baselines match the neural model. Neural-transition claim is killed.
4. **Within-dataset-only result:** held-subject result passes but external dataset fails. Generalization claim is killed.
5. **Calibrated null:** evaluator is valid but no model beats the chief comparator. Publishable limit; model thesis is killed.
6. **Invalid experiment:** leakage, data, calibration, DDP, or evidence audit fails. No scientific result may be reported.

The phrase "the atlas is still useful" must not convert every result into a success. Only classes 1 and 2 support predictive-model claims. Classes 3-5 support bounded methodological or negative findings. Class 6 supports no finding.

---

## 8. Biological and Neurological Foundation

### 8.1 What Scalp EEG Measures

Scalp EEG measures voltage differences created by aggregate extracellular fields, primarily arising from synchronized transmembrane currents across neuronal populations. Synaptic currents are major contributors, while spikes, ionic currents, intrinsic membrane dynamics, geometry, tissue conductivity, and population synchrony also shape the signal. EEG is therefore a lossy, reference-dependent observation of distributed neural activity, not a direct readout of single neurons or a unique cortical source. See Buzsaki, Anastassiou, and Koch's review of extracellular fields and currents: [Nature Reviews Neuroscience](https://pubmed.ncbi.nlm.nih.gov/22595786/).

Consequences for Kahlus:

- a latent variable is not automatically a biological state;
- scalp channels are mixtures of sources through volume conduction;
- referencing changes the observed signal;
- high-frequency and deep activity may be attenuated or poorly recoverable;
- spatial geometry and missing channels must be modeled explicitly;
- sensor-space prediction is valid without claiming source recovery;
- invasive parity is not a valid default target.

### 8.2 Sleep as a Dynamical System

Sleep is organized through interacting cortical, thalamic, hypothalamic, brainstem, autonomic, homeostatic, and circadian processes. Characteristic oscillations include slow waves, spindles, theta activity, and state-dependent changes in cortical and autonomic coupling. Sleep-wake switching is often described using mutually inhibitory sleep-promoting and arousal-promoting circuitry, while REM/NREM transitions involve additional neuromodulatory systems. Reviews include [oscillating circuitries in the sleeping brain](https://pubmed.ncbi.nlm.nih.gov/31616106/) and [sleep state switching](https://pubmed.ncbi.nlm.nih.gov/21172606/).

Why sleep is the first endpoint:

- full-night recordings provide long continuous histories;
- each subject contributes multiple transitions;
- transitions are dense enough for subject-level inference;
- public datasets provide EEG and hypnograms;
- the task is lower risk than retrospective seizure warning;
- coarse Wake/NREM/REM states are more compatible across scoring systems than fine stages;
- duration and circadian priors can be represented by a strong semi-Markov baseline.

### 8.3 Sleep Labels Are Operational Targets

Manual hypnograms are expert annotations, not direct ground truth for a unique latent neural state. A meta-analysis reported substantial overall agreement but much lower agreement for N1 and only moderate reliability for several other stages: [interrater reliability meta-analysis](https://pubmed.ncbi.nlm.nih.gov/34310277/).

Therefore:

- the Phase 0 ontology uses Wake, NREM, REM, and Unknown;
- N1-N3 or R&K stages 1-4 map to NREM for the primary endpoint;
- Unknown, movement, and unscored epochs censor targets;
- a stable transition requires a preregistered run length;
- sensitivity analyses vary the stability definition but cannot replace the primary result;
- claims refer to forecasting the operational hypnogram transition, not discovering the exact biological moment of state change.

### 8.4 Central-Autonomic Coupling

Sleep includes coordinated changes in EEG, heart rate, heart-rate variability, respiration, EOG, EMG, and arousal. Central-autonomic coupling is biologically relevant, but these modalities must not enter the Phase 0 primary model because they would change the question and create additional alignment and missingness risks. The relationship can be investigated later through incremental-information ablations. See the review of [central and autonomic coupling during sleep](https://pubmed.ncbi.nlm.nih.gov/29608990/).

### 8.5 Epilepsy as a Later Stress Test

Epilepsy motivates the long-term need for rare-event forecasting, but public seizure data are sparse, heterogeneous, and vulnerable to patient leakage and recording-context confounding. A 2024 systematic review found that only a small fraction of surveyed seizure-prediction work used patient-independent validation: [cross-patient seizure-prediction review](https://pubmed.ncbi.nlm.nih.gov/39580818/).

CHB-MIT and Siena may be used only after the sleep protocol is stable, initially as observational stress tests. Retrospective event-stratified performance does not authorize the phrases "seizure warning," "seizure predictor," or "clinical monitor."

### 8.6 Source Reconstruction as a Later Validation Lane

Source reconstruction and scalp-to-intracranial mapping are important but active research areas. Recent work has incorporated cortical geometry into source imaging: [Nature Biomedical Engineering, 2026](https://www.nature.com/articles/s41551-026-01664-0). A 2025 concurrent scalp/intracranial study reported strongest predictability at low frequencies and modest explained variance in cross-patient analysis: [scalp EEG predicts intracranial activity](https://pubmed.ncbi.nlm.nih.gov/40291696/).

Kahlus may later test whether a state learned under HNPH predicts source-space or intracranial spectral summaries. This is validation, not the v1 identity and not an implant-equivalence claim.

---

## 9. Measurement and Signal-Processing Principles

### 9.1 Continuous-Recording-First Contract

The atomic scientific unit is a continuous recording with physical metadata, not a pre-windowed tensor.

Each admitted recording must define:

- dataset and source version;
- subject or patient key;
- session/night/run key;
- recording identifier;
- acquisition start and duration;
- native sampling rate;
- physical units;
- channel names;
- reference or lead definition;
- channel geometry source;
- valid intervals and quality flags;
- annotation clock and provenance;
- raw-file checksum status;
- access and license class.

### 9.2 Physical Time

All task definitions use seconds, never sample indices across datasets. Sample indices are derived only after the native sampling rate and resampling transform are validated.

### 9.3 Causality

Input preprocessing must use only information available at or before the forecast issue time. Prohibited operations include:

- zero-phase filters applied across future samples;
- full-night normalization used for a causal input;
- artifact thresholds fitted using test targets;
- target-window interpolation into the context;
- centered temporal convolutions that expose future samples;
- target-derived channel selection;
- retrospective recalibration on test outcomes.

### 9.4 Referencing and Geometry

A channel name alone is insufficient. The protocol records whether a signal is monopolar, linked-mastoid, average-referenced, bipolar, or otherwise derived. When geometry is unavailable or ambiguous, the record must be masked, stratified, or excluded. The system must not silently invent channel coordinates.

### 9.5 Resampling

Resampling requires:

1. verified native rate;
2. an anti-alias filter appropriate to the target rate;
3. versioned transform parameters;
4. a filter guard included in overlap audits;
5. no upsampling presented as new information;
6. validation that annotation clocks remain aligned.

### 9.6 Artifacts and Missingness

EOG, EMG, movement, ECG, electrode drift, clipping, flatlines, impedance changes, line noise, and amplifier saturation can dominate EEG predictions. Missing or invalid samples remain explicitly masked. Zero is a possible voltage and must not represent missingness.

Artifact handling must be reported as part of the data-generating process. A model that predicts artifact onset may be useful for signal-quality monitoring but cannot be presented as forecasting neural state without artifact-stratified evidence.

---

## 10. Formal Mathematical Contract

### 10.1 Recording and Lead Representation

For recording (r), let

\[
X_r(t, \ell) \in \mathbb{R}
\]

denote the measured voltage at physical time (t) for lead or channel \(\ell\). Let \(G_r(\ell)\) encode the lead geometry and reference, and let

\[
M_r(t, \ell) \in \{0,1\}
\]

be the observation and quality mask.

A declared preprocessing operator \(P\) produces a causal, band-limited, resampled representation:

\[
\widetilde{X}_r = P(X_r; \theta_P),
\]

where \(\theta_P\) is fixed before test evaluation and recorded in the evidence bundle.

### 10.2 Forecast Anchor

For issue time \(t\), context duration \(L\), lead \(h\), and target width \(W\):

\[
\mathcal{C}_{r,t} = [t-L,t),
\qquad
\mathcal{T}_{r,t,h} = [t+h,t+h+W).
\]

The required firebreak is:

\[
\mathcal{C}_{r,t} \cap \mathcal{T}_{r,t,h} = \varnothing.
\]

Scored test anchors must also have disjoint raw supports after including filter guards and target intervals. Training anchors may be denser within training subjects, but statistical inference never treats overlapping anchors as independent people.

### 10.3 Waveform-Scale Target

The waveform target is a strictly future signal or complex time-frequency representation:

\[
Y^{\text{wave}}_{r,t,h} = \Phi_{\text{wave}}
\left(\widetilde{X}_r\big|_{\mathcal{T}_{r,t,h}}\right).
\]

Default diagnostic leads are proposed as:

\[
\mathcal{H}_{\text{wave}} = \{0.25, 0.5, 1, 2\}\ \text{seconds}.
\]

This target exists to expose autocorrelation, smoothing, and causality bugs. It is not the headline biological endpoint.

### 10.4 State-Scale Target

The state target is a preregistered multiscale summary computed only from the future target interval:

\[
Y^{\text{state}}_{r,t,h} = \Phi_{\text{state}}
\left(\widetilde{X}_r\big|_{\mathcal{T}_{r,t,h}},G_r,M_r\right).
\]

Candidate components include:

- robust log power in declared frequency bands;
- aperiodic slope and intercept where the rate supports estimation;
- complex spectral coefficients or phase summaries;
- spatial covariance or local graph-edge coherence;
- signal-quality summaries reported separately from neural targets.

The exact \(\Phi_{\text{state}}\) must be frozen before test evaluation. It must not be changed because one representation produces a more favorable result.

Proposed state leads are:

\[
\mathcal{H}_{\text{state}} = \{4,8,16,32\}\ \text{seconds}.
\]

### 10.5 Predictive Distribution

The model outputs a distribution, not only a point estimate:

\[
Q_\theta(Y \mid X_{[t-L,t)},G,M,h).
\]

The distribution may be represented by a low-rank Student-t family, quantiles, or samples, provided that the same proper-score interface evaluates all competitors.

### 10.6 Baseline Residual Form

Let \(B^*\) be the strongest baseline selected using training and validation subjects only. Define the innovation:

\[
R_{r,t,h} = Y_{r,t,h} - \mu_{B^*}(X_{r,t},h).
\]

Kahlus may model the residual distribution:

\[
Q_\theta(Y\mid X,h) = \mu_{B^*}(X,h) + q_\theta(R\mid X,h).
\]

This makes simple continuation the zero-residual reference rather than an easy opponent.

### 10.7 Proper Distributional Score

For predictive samples \(Y,Y' \sim Q\) and observation \(y\), the multivariate energy score is:

\[
\operatorname{ES}(Q,y)
= \mathbb{E}\|Y-y\|_2
- \frac{1}{2}\mathbb{E}\|Y-Y'\|_2.
\]

Proper scoring rules reward both calibration and sharpness. Energy score alone may have limited sensitivity to dependence, so a variogram score and marginal calibration diagnostics should be reported as complementary measures. See [variogram-based proper scoring rules](https://journals.ametsoc.org/view/journals/mwre/143/4/mwr-d-14-00269.1.xml).

### 10.8 Skill Score

For subject \(s\) and lead \(h\):

\[
\operatorname{Skill}_s(h)
= 1 -
\frac{\operatorname{ES}(Q_\theta,Y_s^h)}
{\operatorname{ES}(Q_{B^*},Y_s^h)}.
\]

Subject means are computed before population aggregation. Recording hours do not give one subject more inferential weight than another unless a separate estimand explicitly requires it.

### 10.9 State Predictability Horizon

Define the declared state frontier:

\[
H_{\text{state}}^*
= \max\left\{h \in \mathcal{H}_{\text{state}}:
\forall d \in \mathcal{H}_{\text{state}}\ \text{with}\ d \le h,
\operatorname{LCB}_{95}\left[
\mathbb{E}_s\operatorname{Skill}_s(d)
\right] > \delta,
\ \operatorname{CalPass}(d)
\right\}.
\]

The practical margin \(\delta\) must be preregistered before the sealed external evaluation. It is not a biological constant. The default proposed margin is 0.02 energy-score skill, subject to statistical and domain-expert review during Sprint H0.

The state-scale model claim requires positive skill at two consecutive leads at or beyond 4 seconds on both held-subject and external-dataset evaluations.

### 10.10 Sleep Macrostate Ontology

For scored 30-second epoch \(e\):

\[
S_e \in \{\text{Wake},\text{NREM},\text{REM},\text{Unknown}\}.
\]

R&K stages 1-4 and AASM stages N1-N3 map to NREM for the primary Phase 0 target. This mapping is an operational compatibility choice, not a claim that NREM is homogeneous.

### 10.11 Stable Transition

At issue time \(t\), let the current macrostate be \(S_{t^-}\). A destination \(k \ne S_{t^-}\) is stable when it persists for \(m\) consecutive epochs. The proposed primary value is \(m=2\), corresponding to 60 seconds.

The first stable transition time is:

\[
\tau_t = \inf\{j\Delta:
S_{t+j\Delta}=S_{t+(j+1)\Delta}=k \ne S_{t^-}\},
\quad \Delta=30\text{ s}.
\]

One-epoch and three-epoch definitions are sensitivity analyses only.

### 10.12 Competing-Risk Transition Distribution

For future bin \(j\) and destination \(k\), the model predicts cause-specific hazard:

\[
\lambda_{j,k}
= P(J=j,K=k\mid J\ge j,H_t,Z_t),
\qquad
\sum_k\lambda_{j,k}<1.
\]

Here \(H_t\) is causal EEG history and \(Z_t\) contains allowed causal state variables shared with the chief baseline, such as current macrostate, current bout age, elapsed recording time, and recent transition count.

Survival and event mass are:

\[
S_j=\prod_{r=1}^{j}
\left(1-\sum_k\lambda_{r,k}\right),
\qquad
p_{j,k}=S_{j-1}\lambda_{j,k}.
\]

### 10.13 Censoring-Aware Likelihood

For observed event \((j,k)\), censoring after bin \(c\), or no transition through the horizon:

\[
\mathcal{L}_t =
\begin{cases}
-\log p_{j,k}, & \text{event at }(j,k),\\
-\log S_c, & \text{right-censored after }c,\\
-\log S_J, & \text{no transition through final bin }J.
\end{cases}
\]

This likelihood trains transition probabilities directly. There is no decorative uncertainty head.

### 10.14 Relative Forecast Skill in Bits

For model likelihood contribution \(L_\theta(O_t)\) and chief-baseline likelihood \(L_b(O_t)\):

\[
\operatorname{RFS}
= \frac{1}{N}\sum_t
\log_2\frac{L_\theta(O_t)}{L_b(O_t)}.
\]

RFS is first aggregated within subject, then across subjects.

### 10.15 Transition Lead Bands

The proposed preregistered bands are:

\[
B_1=(0.5,2],\quad
B_2=(2,5],\quad
B_3=(5,10],\quad
B_4=(10,20]\ \text{minutes}.
\]

For a band \(B\), define a categorical forecast over destination states plus a no-event-in-band class \(0\):

\[
q_{\theta,B}(k)=\sum_{j:\,j\Delta\in B}p_{j,k},
\qquad
q_{\theta,B}(0)=1-\sum_k q_{\theta,B}(k).
\]

The held-out evaluator assigns \(O_t^B=k\) when the next stable transition occurs within \(B\) and has destination \(k\); otherwise \(O_t^B=0\). Thus no-transition anchors, later transitions, and earlier transitions all contribute to false-positive control rather than disappearing from the denominator. An anchor censored before the end of \(B\) must not be labeled as class \(0\): it is either excluded under a frozen complete-follow-up rule or handled by a preregistered censoring-aware score. Phase 0 defaults to complete-follow-up anchors for each reported band. For subject \(s\), the band-specific gain over the frozen chief baseline is:

\[
G_s(B)=\frac{1}{|\mathcal{A}_s|}
\sum_{t\in\mathcal{A}_s}
\log_2
\frac{q_{\theta,B}(O_t^B)}{q_{b,B}(O_t^B)},
\]

where \(\mathcal{A}_s\) is the subject's complete eligible natural-grid anchor set, not an event-enriched subset. Population intervals are computed across the subject-level values \(G_s(B)\).

The primary Phase 0 endpoint is the 2-5 minute band on the sealed external sleep dataset.

The transition frontier is the longest contiguous prefix whose subject-level lower confidence bounds are positive and calibration gates pass:

\[
L^* = \max\left\{\sup B_m:
\forall i\le m,
\operatorname{LCB}_{95}\left[\mathbb{E}_sG_s(B_i)\right]>0
\land \operatorname{CalPass}(B_i)
\right\}.
\]

### 10.16 Calibration

Required diagnostics include:

- integrated Brier score;
- calibration intercept and slope;
- integrated calibration index (ICI);
- prediction-set coverage and size;
- PIT or randomized PIT diagnostics where appropriate;
- sharpness or interval width;
- calibration stratified by transition proximity;
- calibration stratified by dataset, montage, signal quality, and current state.

Calibration conditioned on future transition distance is computed only by the held-out evaluator after predictions are frozen. Future labels must never enter model inputs, preprocessing, anchor selection, calibration fitting, or hyperparameter selection.

Conformal methods may be evaluated, but guarantees under dependence and distribution shift must be stated narrowly. Relevant time-series work includes [Conformal Prediction for Time Series](https://pubmed.ncbi.nlm.nih.gov/37819805/). External-dataset calibration must be measured, not assumed.

### 10.17 Multiplicity and Confidence Intervals

- Subjects are the primary bootstrap unit.
- Repeated anchors are nested within subject.
- Simultaneous intervals across horizons use a max-statistic or another preregistered family-wise procedure.
- Test-set hyperparameter selection is prohibited.
- Seed variation is reported separately from subject uncertainty.
- A confidence interval over flattened windows is not an acceptable population interval.

---

## 11. Chief Adversaries and Baseline Ladder

The experiment is designed to defeat named adversaries, not merely populate a comparison table.

### 11.1 Named Adversary A: Autocorrelation and Continuity

Controls:

- persistence;
- last-window repetition;
- smoothed persistence;
- damped harmonic extrapolation;
- delta-target and innovation targets;
- non-overlapping input-target intervals;
- time-shift and target-permutation controls.

### 11.2 Named Adversary B: Linear Dynamical Sufficiency

Chief waveform/state comparators:

- ridge regression;
- autoregressive ridge;
- per-channel AR;
- multichannel VAR where numerically feasible;
- linear Gaussian state-space/Kalman model;
- Koopman-style linear latent dynamics where implementation is reproducible;
- geometry-aware Gaussian-process or linear spatial baseline where feasible.

The model-superiority claim is dead if a compact linear dynamical model matches Kahlus within the preregistered uncertainty interval at the leads that define the scientific claim.

### 11.3 Named Adversary C: State-Duration and Circadian Priors

Chief transition comparator:

- semi-Markov competing-risk model using current macrostate;
- time already spent in the current state;
- elapsed recording time or time of night;
- recent transition count;
- empirical destination and base-rate model.

The neural model must beat this comparator. Beating a no-information or majority-state baseline is insufficient.

### 11.4 Named Adversary D: Acquisition and Subject Identity

Controls:

- subject/session/dataset nuisance predictor;
- unrelated-time context from the same recording;
- shuffled subject blocks;
- geometry permutation;
- channel-order permutation;
- dataset/site probe on learned representations;
- explicit prohibition of IDs, paths, dates, and source keys in model tensors.

### 11.5 Neural Baselines

- small MLP on frozen features;
- causal TCN;
- GRU/LSTM;
- TinySSM or diagonal SSM;
- Transformer with causal mask;
- selective SSM/Mamba-style model only after dependency and kernel reproducibility are proven;
- published sleep forecaster when exact code and protocol are available, otherwise a clearly labeled approximation;
- EEG foundation-model encoder plus a frozen or budget-matched forecasting head where licensing and input contracts permit.

Structured and selective SSMs are candidates because they model long sequences efficiently, not because their names establish novelty. See the [Mamba paper](https://openreview.net/forum?id=AL1fq05o7H) and current EEG-foundation-model limitations in the [2026 critical review](https://pubmed.ncbi.nlm.nih.gov/41666566/).

### 11.6 Baseline Fairness Rules

Every comparator must receive:

- identical train/validation/test subjects;
- identical anchors and masks;
- identical target definitions;
- identical causal metadata allowance;
- documented tuning budget;
- validation-only model selection;
- the same final test opening;
- the same subject-level scoring code.

The flagship model may not train for 50,000 steps while neural baselines receive five smoke steps. Smoke tests and scientific comparisons are separate artifacts.

---

## 12. Data Strategy

### 12.1 Ground-Truth Source Policy

Official repositories and original publications are canonical. Kaggle or other mirrors may aid discovery or transfer but may not define provenance, version, license, or citation.

Raw public neural data must remain outside git. Cluster data caches must be retained for reproducible future runs and must not be deleted by cleanup scripts.

### 12.2 Phase 0 Development Dataset: Sleep-EDF Expanded

[Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/) contains polysomnographic recordings and hypnograms and supports person-grouped sleep-transition development. Both nights from one person must remain in the same split.

Use:

- training and validation;
- held-subject internal test;
- waveform/state evaluator development;
- target ontology and event-count audit;
- calibration development;
- model and baseline selection.

### 12.3 Phase 0 Sealed External Dataset: CAP Sleep Database

The [CAP Sleep Database](https://physionet.org/content/capslpdb/1.0.0/) contains 108 polysomnographic recordings with multiple EEG channels and reference sleep-stage/CAP annotations across healthy and sleep-disorder contexts.

Use:

- untouched external evaluation;
- dataset-shift and montage-shift stress;
- external transition calibration;
- reciprocal replication after the primary direction is frozen.

CAP is not automatically a "site-held-out" dataset unless site provenance supports that wording. Dataset holdout and site holdout are different claims.

### 12.4 Later Sleep Datasets

Candidates after Phase 0:

- Sleep Heart Health Study and other NSRR cohorts, subject to access agreements;
- ANPHY-Sleep high-density EEG/PSG;
- independent prospective sleep-lab cohorts;
- a second sealed institution held outside model development.

Dataset inclusion requires a completed provenance, license, subject-identity, montage, reference, sampling-rate, annotation, and checksum audit.

### 12.5 Epilepsy Stress Datasets

- [CHB-MIT Scalp EEG](https://physionet.org/content/chbmit/1.0.0/);
- [Siena Scalp EEG](https://physionet.org/content/siena-scalp-eeg/1.0.0/);
- TUH/TUSZ after required access and version controls.

CHB-MIT requires explicit handling of the chb01/chb21 duplicate-subject relationship. Siena annotation parsing must be tested against its seizure-list format; an EDF-only fallback is acceptable only for non-event forecasting and must disable seizure-annotation claims.

### 12.6 General EEG Robustness Dataset

[EEGMMI](https://physionet.org/content/eegmmidb/1.0.0/) may test acquisition/task distribution shift, geometry handling, and event-independent state forecasts. EEGMMI is not evidence for sleep or epilepsy transition forecasting.

### 12.7 Dataset Registry Requirements

Each registry entry must include:

- canonical dataset ID and version;
- official source URL;
- original citation and DOI where available;
- access and license class;
- expected size;
- subject, session, and recording identity rules;
- known duplicate-subject issues;
- annotation format;
- channel and reference conventions;
- native sampling rates;
- raw and prepared path policy;
- raw checksum status;
- download date and downloader version;
- allowed tasks;
- prohibited claims;
- known parser or quality issues.

### 12.8 Data Retention Policy

Cluster raw and prepared datasets are reusable research assets.

- Fetch scripts must be idempotent and checksum-aware.
- Successful downloads must not be deleted by run cleanup.
- Run outputs and checkpoints may have retention policies distinct from raw data.
- Any destructive data command requires explicit operator confirmation and a path-scope check.
- The handoff README must state: **Do not delete the shared raw or prepared dataset cache.**

---

## 13. Split, Leakage, and Test-Sealing Protocol

### 13.1 Split Before Transformation

Subject, session, site, and dataset assignments occur before:

- filtering;
- normalization;
- resampling statistics;
- artifact-threshold fitting;
- target extraction;
- window or anchor generation;
- augmentation;
- representation learning;
- calibration;
- hyperparameter search.

### 13.2 Required Split Regimes

1. Subject-held-out within each dataset.
2. Dataset-held-out external evaluation.
3. Site-held-out only when genuine independent site labels exist.
4. Chronologically purged support/query splits for later personalization.

### 13.3 Test Seal

Before opening the external test:

- source-file manifest is frozen and hashed;
- subject grouping is frozen;
- target ontology is frozen;
- lead bands are frozen;
- baseline family is frozen;
- model family and tuning budget are frozen;
- calibration method is frozen;
- report template is frozen;
- software commit and environment are recorded;
- a red-team reviewer signs the readiness checklist.

After the seal, external results may be computed once for the preregistered model family. Any architecture change creates a new protocol version and new external test set or explicitly exploratory status.

### 13.4 Transition-Label Firewall

Future sleep or seizure labels must not influence:

- context normalization;
- artifact thresholds;
- anchor inclusion except through the frozen target/censoring contract;
- sample weighting in evaluation;
- model input tensors;
- representation pretraining on test records;
- calibration fitting;
- baseline selection;
- checkpoint selection.

Post-hoc transition-distance strata are computed only after predictions and the primary score packet are frozen.

---

## 14. Proposed Software Architecture

All types and signatures in this section are **proposed contracts**, not claims about the current implementation.

### 14.1 Core Types

```python
@dataclass(frozen=True)
class LeadGeometry:
    lead_id: str
    positive_xyz_m: tuple[float, float, float] | None
    negative_xyz_m: tuple[float, float, float] | None
    reference_kind: str
    position_source: str


@dataclass(frozen=True)
class PhysicalSignalRecord:
    record_id: str
    subject_id: str          # evaluator and split only
    session_id: str
    dataset_id: str
    site_id: str | None
    sampling_rate_hz: float
    physical_unit: str
    leads: tuple[LeadGeometry, ...]
    raw_source_uri: str
    source_sha256: str | None
    valid_intervals_s: tuple[tuple[float, float], ...]
    annotation_uri: str | None


@dataclass(frozen=True)
class ForecastAnchor:
    record_id: str
    context_start_s: float
    context_end_s: float
    target_start_s: float
    target_end_s: float
    lead_s: float
    filter_guard_s: float

    def validate_non_overlap(self) -> None:
        """Raise if causal context, guard, or target intervals overlap."""


@dataclass(frozen=True)
class StateTargetSpec:
    version: str
    bands_hz: tuple[tuple[float, float], ...]
    target_window_s: float
    include_aperiodic: bool
    include_spatial_covariance: bool
    include_complex_spectrum: bool


@dataclass(frozen=True)
class TransitionTarget:
    event_bin: int | None
    destination: str | None
    censored_after_bin: int | None
    no_transition_through_horizon: bool
    ontology_version: str


@dataclass(frozen=True)
class ModelBatch:
    signal: torch.Tensor
    observed_mask: torch.Tensor
    artifact_mask: torch.Tensor
    lead_geometry: torch.Tensor
    lead_mask: torch.Tensor
    relative_time_s: torch.Tensor
    horizon_s: torch.Tensor
    allowed_context: dict[str, torch.Tensor]
    # Subject, session, dataset, site, path, and future labels are absent.


@dataclass(frozen=True)
class ForecastDistribution:
    mean: torch.Tensor
    scale: torch.Tensor
    low_rank_factor: torch.Tensor | None
    degrees_of_freedom: torch.Tensor | None
    samples: torch.Tensor | None


@dataclass(frozen=True)
class TransitionDistribution:
    hazards: torch.Tensor
    event_mass: torch.Tensor
    survival: torch.Tensor
    ensemble_disagreement: torch.Tensor | None


@dataclass(frozen=True)
class EvidenceDecision:
    protocol_version: str
    gate_passed: bool
    outcome_class: str
    failed_requirements: tuple[str, ...]
    allowed_claims: tuple[str, ...]
    blocked_claims: tuple[str, ...]
```

### 14.2 Proposed Functions

```python
def build_recording_registry(...) -> RecordingRegistry: ...
def build_frozen_split_bundle(...) -> SplitBundle: ...
def materialize_forecast_anchors(...) -> ForecastAnchorIndex: ...
def compute_state_targets(...) -> StateTargetStore: ...
def compute_transition_targets(...) -> TransitionTargetStore: ...
def audit_forecast_firebreak(...) -> AuditReport: ...
def run_frozen_baseline_ladder(...) -> BaselineEvidence: ...
def train_probabilistic_forecaster(...) -> TrainedRun: ...
def fit_validation_calibrator(...) -> FrozenCalibrator: ...
def score_subject_level_horizons(...) -> HorizonEvidence: ...
def score_transition_frontier(...) -> TransitionEvidence: ...
def build_final_evidence_decision(...) -> EvidenceDecision: ...
```

These are design signatures. Implementations must not be added as empty no-op stubs. If a scaffold is committed before implementation, it must raise `NotImplementedError` and be labeled `PROPOSED` or `STUB` in code and documentation.

### 14.3 Proposed Module Map

```text
src/neurotwin/forecastability/
  contracts.py
  registry.py
  lead_geometry.py
  causal_preprocessing.py
  splits.py
  anchors.py
  state_targets.py
  transition_targets.py
  baselines.py
  controls.py
  scoring.py
  calibration.py
  bootstrap.py
  audits.py
  gates.py
  reports.py

src/neurotwin/models/forecastability/
  config.py
  lead_tokenizer.py
  geometry_encoder.py
  temporal_backbone.py
  state_distribution.py
  transition_hazard.py
  model.py

src/neurotwin/training/
  forecast_command.py
  forecast_loop.py
  forecast_metrics.py
  forecast_checkpoint.py

src/neurotwin/adapters/
  sleep_edf_frontier.py
  cap_sleep_frontier.py
```

The dominant call path should remain short:

```text
adapter -> registry/split -> anchors/targets -> baseline or model
       -> frozen calibration -> subject-level scoring -> gate/report
```

### 14.4 Relationship to Existing Modules

Preserve and extend:

- `src/neurotwin/data/split_manifest.py`;
- `src/neurotwin/data/audit.py`;
- `src/neurotwin/data/leakage.py`;
- `src/neurotwin/adapters/multidataset.py` where contracts remain valid;
- `src/neurotwin/benchmarks/baseline_suite.py`;
- `src/neurotwin/eeg_v1/` as historical baseline and migration source;
- `src/neurotwin/eval/paper_gate.py`;
- `src/neurotwin/reports/` evidence writers;
- `src/neurotwin/runtime/distributed.py` after the lockstep fix is integrated;
- A100 packaging and checksum infrastructure.

Demote or archive as scientific center:

- current overlapping future-window evaluator;
- generic five-task neural-translation headline;
- NFC as presumed flagship;
- KTM, dual-field, and Transition Gym as human evidence;
- source-to-intracranial parity framing;
- broad multimodal and clinical claims.

NFC, KTM, dual-field, and Transition Gym may remain as baselines, synthetic invariance tests, or future experimental branches. They do not define the Phase 0 claim.

---

## 15. Machine-Learning Architecture

### 15.1 Minimum Phase 0 Model

The first Kahlus model should be a 20-40 million parameter causal geometry-aware probabilistic forecaster, not a billion-parameter foundation model.

Components:

1. **Causal lead tokenizer**
   - consumes microvolt-valued signal patches;
   - carries sample and channel masks;
   - records physical time and patch duration;
   - never sees future target samples.

2. **Geometry-aware lead-set encoder**
   - treats leads as a set or graph;
   - supports arbitrary observed lead subsets;
   - uses measured or declared standard coordinates;
   - masks unknown geometry rather than guessing;
   - does not encode dataset or subject IDs.

3. **Multiscale causal temporal backbone**
   - small GRU/SSM first;
   - causal TCN and Transformer as matched alternatives;
   - selective SSM only after baseline and dependency checks;
   - explicit physical-time and horizon conditioning.

4. **State distribution head**
   - predicts mean and trained uncertainty;
   - low-rank covariance or sample-based distribution;
   - masked likelihood or proper score;
   - no untrained report-only uncertainty.

5. **Competing-risk transition head**
   - predicts event-time bins and destination state;
   - emits survival and event mass;
   - trains with censoring-aware likelihood;
   - receives only causal state variables allowed to the chief baseline.

6. **Validation-only calibrator**
   - fitted after model selection;
   - frozen before external test;
   - never refitted on external labels.

### 15.2 Multi-Task Relationship

State and transition tasks may share an encoder only after separate task implementations pass independently. Joint training is allowed only if:

- loss weights are declared and tuned without external test access;
- the shared encoder does not receive future labels;
- both tasks retain independent heads and score packets;
- transition labels do not alter state-task anchor selection;
- ablations compare joint and separate training fairly.

The current prepared pipeline trains separate models per task. It must not be described as one jointly trained multimodal latent-state model until the software actually implements that behavior.

### 15.3 Losses

Candidate state loss:

\[
\mathcal{L}_{\text{state}}
= \mathcal{L}_{\text{NLL}}
+ \lambda_{\text{ES}}\mathcal{L}_{\text{energy}}
+ \lambda_{\text{var}}\mathcal{L}_{\text{variogram}},
\]

with all terms masked to observed valid targets.

Candidate transition loss:

\[
\mathcal{L}_{\text{transition}}
= -\sum_i \log L_\theta(O_i)
+ \lambda_{\text{cal}}\mathcal{R}_{\text{calibration}},
\]

where any calibration regularizer is supplemental to the proper event likelihood.

Training loss and final evaluation score need not be identical, but deviations require justification and ablation.

### 15.4 Model-Selection Rules

- Tuning occurs on validation subjects only.
- The tuning budget is equalized or transparently cost-adjusted.
- Checkpoint selection uses the preregistered validation score.
- Test metrics never drive early stopping.
- Five seeds are recommended for the locked final family.
- Subject bootstrap uncertainty is not replaced by seed variance.
- Ensemble probability averaging is allowed if preregistered.
- Model size and compute are reported.

### 15.5 Required Ablations

- remove geometry;
- permute geometry;
- remove EEG history while retaining state-duration priors;
- remove state-duration priors while retaining EEG;
- remove uncertainty training;
- replace temporal backbone with linear dynamics;
- remove cross-lead mixing;
- channel dropout and low-density subsets;
- context lengths;
- state target components;
- joint versus separate state/transition training;
- external calibration sensitivity;
- artifact-heavy versus artifact-light strata.

---

## 16. Statistical Analysis Plan

### 16.1 Estimands

Primary transition estimand:

> Mean subject-balanced incremental log skill in the 2-5 minute lead band on the sealed CAP external dataset relative to the strongest validation-selected comparator.

Primary state estimand:

> Mean subject-balanced energy-score skill at two consecutive state leads at or beyond 4 seconds on held-subject Sleep-EDF and sealed external data.

### 16.2 Subject Weighting

Each subject contributes one aggregate per lead or lead band. A subject with a longer recording does not automatically receive greater population weight. Anchor-level metrics remain available for diagnostic and within-subject analyses.

### 16.3 Bootstrap

- outer resampling unit: subject;
- optional inner resampling: non-overlapping anchors within subject;
- confidence level: 95%;
- one-sided lower bound for preregistered superiority gates;
- simultaneous correction across lead bands;
- fixed random seeds recorded in evidence.

### 16.4 Event Count Audit

Before sealing the external test but without training a neural model:

- count eligible subjects;
- count stable transitions by source and destination;
- count events in each lead band;
- count censored anchors;
- estimate whether the primary external band has adequate subject support;
- freeze any threshold adjustment before test predictions are generated.

Event-count auditing may alter the protocol only before the test seal. It must not inspect model performance.

### 16.5 Calibration Near Transitions

Report calibration:

- overall;
- by current macrostate;
- by destination;
- by transition lead band;
- near versus far from transitions;
- by signal quality;
- by dataset/montage;
- by pathology class as descriptive analysis only.

A model cannot pass by being calibrated only on the dominant no-transition state.

### 16.6 Missing Data

Missingness may be informative. Required analyses:

- missingness prevalence by dataset and state;
- performance by missing-channel count;
- performance by artifact burden;
- missingness-only nuisance model;
- complete-case sensitivity;
- explicit masking in every likelihood denominator.

### 16.7 Subgroup Analysis

Age, sex, pathology, medication, and other available metadata are descriptive unless a separate powered protocol is preregistered. Small-subgroup results must not be interpreted as fairness, clinical validity, or biological mechanism.

---

## 17. End-to-End Development Roadmap

Every sprint ends with a runnable verification command added by that sprint. Commands below are behavioral requirements, not invented current CLI guarantees.

### Sprint G0: Governance and Repository Alignment

**Goal:** establish a clean, current implementation base.

Tasks:

- reconcile the current worktree with `origin/add-researchdock-roadmap`;
- incorporate the known DDP lockstep fix at commit `733f5fed` or its merged descendant;
- verify no user changes are lost;
- select a dedicated HNPH branch;
- freeze this protocol version;
- identify PI, neuroscience, statistics, engineering, and red-team owners;
- create a decision log and requirement ledger.

Acceptance:

- clean worktree proof;
- exact commit recorded;
- DDP lockstep tests present and passing;
- protocol reviewed before model work;
- no A100 full run launched.

### Sprint H0: Forecastability Contract Freeze

**Goal:** implement a ruler capable of saying no.

Tasks:

- implement physical record, lead geometry, interval, anchor, target, and evidence contracts;
- freeze lead grids and transition ontology;
- implement non-overlap and filter-guard audits;
- define chief comparators and tuning budgets;
- define subject-level bootstrap and calibration gates;
- implement synthetic fixtures that intentionally violate each rule.

Acceptance:

- overlap-injection test fails closed;
- future-normalization test fails closed;
- forbidden-identity-field test fails closed;
- invalid sampling/unit/reference tests fail closed;
- all proposed protocol thresholds are either frozen or explicitly marked unresolved before data opening.

### Sprint H1: Sleep-EDF and CAP Physical Adapters

**Goal:** produce trusted continuous-record registries.

Tasks:

- parse EDF signals and annotations;
- group nights by person;
- validate physical units and rates;
- map lead geometry and references;
- map R&K/AASM stages to the frozen ontology;
- generate raw-source and transform hashes;
- report exclusions with reasons;
- preserve raw caches.

Acceptance:

- every admitted record has a complete data card;
- all same-person nights remain in one split;
- signal/annotation clocks align within declared tolerance;
- excluded records are enumerated;
- raw data remain outside git.

### Sprint H2: Baseline-Only Horizon Study

**Goal:** establish the honest floor before Kahlus.

Tasks:

- materialize disjoint waveform and state anchors;
- implement persistence, ridge, AR, VAR/Kalman, harmonic, and simple spatial baselines;
- implement semi-Markov and feature-hazard transition baselines;
- run time-shift, target-permutation, unrelated-context, and identity controls;
- emit subject-level curves and calibration.

Acceptance:

- chief baseline selected without external model results;
- negative controls have no positive lower confidence bound;
- all baseline outputs are finite;
- baseline rankings are checksummed;
- data and model budgets are documented.

### Sprint H3: State-Scale Probabilistic Forecaster

**Goal:** test whether non-linear learned dynamics add state-scale information.

Tasks:

- implement the minimal causal geometry-aware model;
- train uncertainty using proper losses;
- compare GRU, causal TCN, TinySSM, Transformer, and proposed model under matched budgets;
- run one-device debug and resume tests;
- fit validation-only calibration;
- perform required state ablations.

Acceptance:

- no target sample enters context computation;
- uncertainty outputs are finite and trained;
- predictions reproduce after checkpoint resume within declared tolerance;
- state gate is evaluated on held-subject data before external test;
- no broad claim if the chief linear baseline ties.

### Sprint H4: Sleep Transition Target and Hazard Models

**Goal:** implement the first biological endpoint.

Tasks:

- implement stable-transition and censoring targets;
- verify event-time and destination calculations by hand-labeled fixtures;
- implement competing-risk likelihood;
- implement oracle semi-Markov chief comparator;
- implement transition-conditioned calibration reports;
- hide future labels from model and preprocessing tensors.

Acceptance:

- likelihood sums to a valid probability distribution;
- survival is monotone and nonnegative;
- censored and no-transition cases are correct;
- label-shift and permutation controls erase neural gain;
- scorer-definition sensitivity is reported.

### Sprint H5: Locked Model Family

**Goal:** select exactly one final Phase 0 family.

Tasks:

- run fixed tuning budget on Sleep-EDF training/validation only;
- select architecture, parameter budget, context length, and calibration method;
- train fixed seeds;
- freeze checkpoints and prediction schema;
- seal CAP manifest and report template.

Acceptance:

- model family frozen before CAP prediction;
- no CAP label or metric informed selection;
- all seeds complete or are explicitly failed;
- calibrator uses validation subjects only;
- signed readiness packet exists.

### Sprint H6: External Phase 0 Evaluation

**Goal:** open the sealed external result once.

Tasks:

- generate CAP predictions;
- compute subject-level state and transition evidence;
- run calibration and negative controls;
- run reciprocal CAP-to-Sleep-EDF replication without changing contracts;
- classify outcome using the frozen gate.

Acceptance:

- external predictions are immutable and checksummed;
- no post-test calibration or tuning occurred;
- primary and all failed endpoints are reported;
- outcome class is machine-readable;
- allowed and blocked claims are explicit.

### Sprint H7: Independent Reproduction and Paper Packet

**Goal:** make the result auditable without model-development context.

Tasks:

- independent scorer reconstructs metrics from prediction tables;
- rerun from clean commit/environment;
- produce figures from source tables;
- create model card, dataset cards, limitations, and claim audit;
- release negative results and failed controls;
- obtain expert review from neuroscience and statistics collaborators.

Acceptance:

- independent metrics match within tolerance;
- figures trace to checksummed tables;
- manuscript numbers trace to artifacts;
- no unsupported clinical language;
- complete evidence archive exists.

### Sprint Y1: Multi-Dataset Retrospective Atlas

Only after a Phase 0 pass or scientifically justified limited result:

- add SHHS/NSRR or other independent cohorts;
- add true center/site holdouts where metadata permit;
- test montage, age, pathology, and scorer shift;
- operate a sealed evaluation service;
- invite external model submissions.

### Sprint Y2: Prospective Multi-Lab Replication

- preregister new full-night recordings;
- freeze model before first test night;
- emit prospective probabilities with timestamps;
- blind scorers to predictions;
- test new hardware, operators, and missing channels;
- publish calibration drift and failures.

### Sprint Y3: Low-Density and Wearable Research Recorder

- determine minimal lead subsets from retrospective ablations;
- validate signal quality and state frontier prospectively;
- add ECG/PPG, IMU, EOG, or respiration only through incremental-information tests;
- implement event-triggered high-resolution storage as a research feature;
- make no diagnosis or warning claim.

### Sprint Y4: Causal Perturbation Study

Only after prospective observational replication:

- define a benign, IRB-approved perturbation such as auditory stimulation;
- randomize timing independently of model prediction where required for identification;
- preregister physiological endpoints;
- test whether forecast strata predict heterogeneous response;
- separate causal effect estimation from observational forecasting.

### Sprint Y5: High-Risk Neurological Replication

Only with clinical collaborators and adequate event counts:

- evaluate seizure or anesthesia transitions under separate governance;
- retain patient-held-out, site-held-out, and prospective calibration;
- do not inherit sleep claims automatically;
- pursue clinical-device development only through a separate regulatory protocol.

---

## 18. Compute and A100 Protocol

### 18.1 Compute Is Not the Scientific Bottleneck

The 7-8 A100 cluster enables seed, ablation, and scale studies. It cannot repair target overlap, invalid splits, poor calibration, weak baselines, or insufficient external data.

### 18.2 Compute Escalation

1. CPU/unit fixtures.
2. Local tiny data and baseline smoke.
3. One GPU import/data/model smoke.
4. One GPU short training and resume.
5. Two-process DDP lockstep test.
6. One A100 scientific debug.
7. Multi-A100 seed/ablation rehearsal.
8. Locked 7xA100 run only after all gates pass.

### 18.3 Preferred Use of Seven GPUs

For Phase 0, independent seed and ablation jobs may be scientifically safer than forcing every run into seven-way DDP. Multi-GPU DDP is justified only when one model cannot fit or complete within the wall-time budget and after numerical parity tests.

### 18.4 DDP Requirements

- exact same optimizer-step count on every rank;
- distributed sampler with explicit epoch/seed behavior;
- consistent `drop_last` policy;
- rank-synchronized nonfinite detection;
- DDP forward path retained during distributed inference where collectives are expected;
- rank-zero-only shared artifact writes;
- all-rank barriers at declared boundaries;
- all-gather or reduce metrics with explicit denominators;
- resume restores sampler, optimizer, scheduler, scaler, and step;
- NCCL timeout and heartbeat configuration documented;
- world size recorded in every metric artifact.

The prior seven-GPU run failed because ranks drifted dramatically in training progress and an ALLREDUCE timed out. Any recurrence invalidates the run; partial rank metrics cannot be combined into a scientific result.

### 18.5 A100 Handoff Bundle

Required:

- exact commit hash;
- clean-worktree proof;
- runner archive and checksum;
- lock/environment files;
- dataset-fetch and audit instructions;
- persistent shared raw/prepared cache policy;
- CPU smoke;
- one-GPU smoke;
- DDP lockstep smoke;
- exact 7xA100 command;
- expected CPU, RAM, disk, and wall time;
- output schema;
- success and failure signatures;
- resume procedure;
- evidence bundle writer;
- no secrets, raw private data, or unauthorized checkpoints.

### 18.6 Expected Resource Envelope

Resource estimates must be produced by the implemented estimator before launch. The protocol does not invent a fixed runtime. A run may use hours or days only after measured throughput, memory, I/O, and convergence estimates justify the allocation.

---

## 19. Verification and Test Matrix

### 19.1 Unit Tests

- interval algebra and non-overlap;
- filter guards;
- sample/second conversion;
- lead geometry parsing;
- physical-unit conversion;
- ontology mapping;
- stable-transition calculation;
- censoring likelihood;
- hazard normalization;
- energy/log/Brier scores;
- subject aggregation;
- bootstrap determinism;
- calibration metrics;
- checksum validation.

### 19.2 Property Tests

- survival is monotone;
- event mass plus terminal survival sums to one;
- masked targets do not affect loss;
- channel permutation with corresponding geometry permutation preserves output;
- adding unavailable channels does not change observed-channel predictions beyond tolerance;
- future samples cannot change past embeddings;
- duplicate records trigger audit failure;
- synthetic time shift destroys skill;
- target permutation destroys skill.

### 19.3 Integration Tests

- tiny Sleep-EDF-like EDF plus annotation fixture;
- tiny CAP-like EDF plus scoring fixture;
- split before anchor generation;
- prepared artifact reload and hash verification;
- baseline and model consume identical anchor IDs;
- calibration fitted on validation only;
- final report built from frozen prediction tables.

### 19.4 Distributed Tests

- two-process CPU/Gloo lockstep;
- two-GPU NCCL smoke where available;
- rank-tagged step metrics;
- checkpoint/resume parity;
- synchronized nonfinite abort;
- no concurrent shared-file corruption;
- final world-size audit.

### 19.5 Adversarial Scientific Tests

- intentionally overlapping input/target;
- centered filter leakage;
- full-record normalization leakage;
- same-subject night split;
- hidden path/ID tensor;
- dataset-probe success;
- transition-label preprocessing leak;
- event-enriched evaluation prevalence;
- window-level bootstrap inflation;
- external-test recalibration;
- baseline undertraining;
- broad-claim injection into report.

---

## 20. Evidence Bundle Contract

Every scientific run must produce:

```text
evidence/
  protocol_snapshot.md
  protocol_hash.json
  git_state.json
  environment.json
  command.json
  config_resolved.yaml
  dataset_registry.json
  dataset_counts.csv
  source_checksums.json
  preprocessing_contract.json
  split_manifest.json
  split_audit.json
  anchor_manifest.json
  overlap_audit.json
  baseline_budget.json
  baseline_rankings.csv
  per_subject_scores.csv
  horizon_curve.csv
  transition_frontier.csv
  calibration.json
  negative_controls.json
  ablations.csv
  failed_runs.json
  prediction_manifest.json
  model_card.md
  data_cards/
  figures/
  logs/
  final_gate.json
  claim_audit.json
  SHA256SUMS
```

Raw public EEG and raw private participant data must not be copied into the evidence bundle.

### 20.1 Success Signature

A run is complete only when:

- every required artifact exists;
- checksums pass;
- world size and rank counts agree;
- no rank reports incomplete training;
- all metrics are finite or explicitly defined missing;
- split and overlap audits pass;
- chief baselines are present;
- negative controls are null;
- calibration report exists;
- final gate and claim audit agree;
- independent scoring reproduction passes.

---

## 21. Known Bugs and Likely Engineering Failures

### 21.1 Current Known Repository Issues

As of this protocol date:

- the active worktree is at `f16b65e7` and behind `origin/add-researchdock-roadmap`;
- remote commit `733f5fed` contains a DDP lockstep fix not present in the active worktree;
- the current scientific future-window path does not implement a true physical multi-horizon evaluator;
- some existing window definitions can overlap input and target when lead is shorter than target width;
- existing site/dataset manifests do not always correspond to runnable held-out evaluation;
- current multidataset harmonization is insufficient for physical cross-dataset comparison;
- current prepared training builds separate models per task rather than one joint model;
- current NFC ignores additional source modalities in important paths and does not train uncertainty under a proper probabilistic objective;
- current confidence intervals are not consistently subject-clustered;
- the earlier seven-GPU run failed from DDP rank drift and NCCL timeout.

These issues must be fixed or bypassed by the new contract before scientific training.

### 21.2 Data and Signal Bugs

| Bug | Observable symptom | Required response |
| --- | --- | --- |
| Wrong EDF physical scaling | Implausible microvolt ranges or dataset-specific amplitudes | Reject record; verify digital/physical min/max |
| Annotation clock offset | Transition labels shifted from traces | Hand-audit landmarks; fail dataset gate |
| Same person in multiple splits | Inflated held-subject score | Canonical identity map; rebuild splits |
| Duplicate mirror files | Apparent external replication | Raw hashes and provenance deduplication |
| Causal filter implemented as zero-phase | Unrealistically strong short-horizon score | Filter impulse/causality test; invalidate run |
| Resampling aliasing | Spurious spectral structure | Anti-alias test and transform audit |
| Channel label mismatch | Geometry-aware model learns wrong topology | Explicit mapping and exclusion report |
| Reference mismatch | Cross-dataset failure or artificial differences | Lead contract and reference strata |
| Missingness encoded as zero | Model confuses missing channels with activity | Separate boolean masks |
| Full-night normalization | Future leakage | Context-only or train-only statistics |
| CAP/Sleep-EDF ontology mismatch | False transition events | Versioned mapping and sensitivity analysis |
| Siena text parser failure | Missing seizure annotations | Fix parser or disable event claim explicitly |
| CHB01/CHB21 identity duplication | Patient leakage | Canonical patient merge before split |

### 21.3 Model and Numerical Bugs

| Bug | Observable symptom | Required response |
| --- | --- | --- |
| Noncausal padding/mask | Future perturbation changes past embedding | Causality property test |
| Hazard sum exceeds one | Negative survival or NaN likelihood | Constrained parameterization and assertions |
| Survival underflow | `-inf` NLL at long horizons | Log-space survival computation |
| Student-t degrees of freedom invalid | NaN sampling or loss | Positive transform and bounded floor |
| Covariance not positive definite | Cholesky failure | Low-rank plus positive diagonal construction |
| Scale inflation | Good coverage, useless broad intervals | Proper score and sharpness gate |
| Scale collapse | Overconfident NLL explosion | Floors, gradient checks, calibration audit |
| Mask denominator zero | NaN batch loss | Explicit empty-target rejection |
| Teacher forcing leaks future | Strong train, failed causal inference | Non-teacher-forced test and API audit |
| Subject embedding leakage | Excellent in-domain, failed external | Remove IDs; representation probe |
| Dataset/montage memorization | High dataset-probe accuracy | External holdout and geometry ablations |
| Mixed precision overflow | Inf gradients/loss | scaler logs, finite checks, fallback precision |
| Resume mismatch | Different metrics after restart | Full-state checkpoint parity test |

### 21.4 Distributed and Cluster Bugs

| Bug | Observable symptom | Required response |
| --- | --- | --- |
| Rank step drift | One rank finishes while others lag | Fixed sampler/steps; synchronized loop |
| Collective mismatch | NCCL timeout | Same collective order on every rank |
| Rank-local early stop | Some ranks exit first | Distributed stop reduction |
| Uneven final batches | Step-count mismatch | Explicit sampler and `drop_last` policy |
| Shared-file races | Corrupt JSON/checkpoints | Rank-zero writes and atomic rename |
| Stale dataset cache | Inconsistent counts/hashes | Versioned cache manifests |
| Data loader I/O starvation | Low GPU utilization | Profile workers, local staging, shard balance |
| GPU count mislabel | Wrong world size | Runtime and evidence assertion; label 7xA100 honestly |
| Container version drift | Nonreproducible evidence | Lockfile/image digest and environment capture |
| Dependency/evidence mismatch | Metrics generated under different versions | Pin or regenerate evidence |
| Accidental cache deletion | Repeated large downloads | Protected persistent raw/prepared directories |

### 21.5 Statistical Bugs

| Bug | Consequence | Required response |
| --- | --- | --- |
| Window-level bootstrap | False precision | Bootstrap subjects |
| Test baseline selection | Optimistic skill | Freeze on validation |
| External recalibration | Invalid external calibration claim | Validation-only calibrator |
| Multiple favorable horizons | Cherry-picked lead | Simultaneous intervals and contiguous frontier |
| Event-enriched test sampling | Miscalibrated prevalence | Natural-grid evaluation |
| Informative censoring ignored | Biased transition risk | Censor likelihood and sensitivity analysis |
| Small destination subgroup | Unstable cause-specific claim | Report support; no subgroup claim |
| Dataset and site conflated | False site-generalization claim | Separate identifiers and wording |
| Seed variance called population uncertainty | Invalid inference | Separate seed and subject uncertainty |
| Negative control omitted | Leakage remains invisible | Gate requires every control |

---

## 22. Blockers and Stop Conditions

### 22.1 Immediate Blockers

- current branch behind the DDP fix;
- no frozen HNPH/KSTF target contract in code;
- no complete physical lead/units/reference model;
- no genuine state-scale horizon evaluator;
- no CAP adapter under the new contract;
- no subject-level probabilistic scoring implementation;
- no external test seal;
- no independent statistical reviewer assigned.

### 22.2 Scientific Blockers

- insufficient external subject/event support in the primary lead band;
- sleep scoring incompatibility that cannot be resolved under coarse ontology;
- no positive incremental information beyond semi-Markov and linear baselines;
- calibration failure under external acquisition shift;
- transition labels too noisy to support the endpoint;
- state target excessively dependent on arbitrary preprocessing;
- representation dominated by artifact or hardware identity.

### 22.3 Operational Blockers

- unavailable data-use approval;
- ambiguous licenses;
- inability to preserve raw data caches;
- insufficient disk for raw plus prepared data;
- cluster environment not reproducible;
- no owner for external test seal;
- no reviewer willing to evaluate the clinical wording.

### 22.4 Stop Conditions

Stop larger-model work when:

- a chief comparator ties Kahlus externally;
- calibration remains outside the frozen gate after validation-only fitting;
- a negative control shows positive skill;
- external gain appears only after retuning;
- the result depends on one montage, one subgroup, or one label mapping;
- repeated architectural additions are being used to avoid a failed hypothesis;
- model size grows before the evaluator is stable;
- a clinical claim is proposed from retrospective evidence.

---

## 23. Risk Register

| ID | Risk | Probability | Impact | Trigger | Mitigation / decision |
| --- | --- | --- | --- | --- | --- |
| R1 | True frontier is short or zero | High | High | Linear model ties at all state leads | Publish bound; kill world-model claim |
| R2 | Transition prior dominates EEG | High | High | Semi-Markov ties externally | Kill neural transition claim |
| R3 | External acquisition shift breaks calibration | High | High | CAP ICI/slope/Brier fail | Narrow claim; no test recalibration |
| R4 | Label uncertainty dominates boundary | Medium | High | Sensitivity changes sign | Redefine future protocol; primary run remains failed |
| R5 | Montage fingerprint drives gain | Medium | High | Dataset probe succeeds | Geometry/missingness redesign; repeat with new seal |
| R6 | DDP rank drift recurs | Medium | High | Unequal final steps | Invalidate run; return to lockstep tests |
| R7 | Dataset parser silently drops records | Medium | High | Counts disagree with source | Fail preparation gate |
| R8 | Baselines are undertrained | Medium | High | Budget/audit mismatch | Rerun all models under fair protocol |
| R9 | Proper score rewards diffuse output | Medium | Medium | Wide intervals with weak sharpness | Add sharpness/variogram and calibration gates |
| R10 | Team chases architecture novelty | High | Medium | Model work before H0/H2 | Governance stop; baseline-first rule |
| R11 | Prospective partners unavailable | Medium | High | No sealed cohort by Year 2 | Retain retrospective scope; no deployment claim |
| R12 | Wearable signal quality is inadequate | Medium | Medium | Low-density frontier collapses | Treat as hardware limit, not software failure |
| R13 | Epilepsy branch becomes overclaimed | High | High | Warning/diagnosis wording appears | Claim gate and expert review |
| R14 | Raw/private data enters git or bundle | Low | Critical | Secret/raw-data audit hit | Block packaging; remove through approved process |
| R15 | Protocol becomes too large to operationalize | Medium | Medium | Requirements not traced to tests | Requirement ledger and sprint extraction |

---

## 24. Team and Governance

### 24.1 Ten-Person Suggested Allocation

- 2 data/provenance and physical-signal owners;
- 2 split, target, and leakage-audit owners;
- 2 baseline and reproduction owners;
- 2 model, training, and DDP owners;
- 1 statistics and calibration owner;
- 1 independent red-team, preregistration, and evidence owner.

Roles may overlap, but the external-test opener and primary model developer should not be the same person without independent oversight.

### 24.2 Required Expert Review

- EEG/neurophysiology reviewer;
- sleep-science reviewer;
- statistical forecasting/calibration reviewer;
- distributed-systems reviewer before A100 launch;
- clinical claim-boundary reviewer before any epilepsy or patient-facing material.

Potential mentors may be invited, but this protocol does not claim their endorsement.

### 24.3 Decision Log

Every decision that changes target, split, metric, threshold, dataset, baseline, model family, or claim must record:

- date;
- decision owner;
- evidence available at the time;
- alternatives considered;
- whether external test had been opened;
- protocol version impact;
- rerun requirements.

---

## 25. Ethics, Privacy, and Regulatory Boundary

### 25.1 Public Retrospective Data

- honor source licenses and data-use terms;
- retain canonical citations;
- do not attempt reidentification;
- do not redistribute raw data outside allowed terms;
- document excluded and transformed fields;
- separate research benchmarking from clinical claims.

### 25.2 Prospective Human Studies

Require:

- IRB approval or formal determination;
- informed consent;
- predefined data retention and sharing;
- adverse-event and privacy procedures;
- prospective registration where appropriate;
- clear non-diagnostic language;
- human-factors review for any wearable.

### 25.3 Clinical Device Boundary

A software or wearable prototype becomes a clinical-development program only after its intended use, target population, output, decision consequence, and risk class are explicitly defined. Retrospective forecast performance does not establish safety, effectiveness, or clinical utility.

---

## 26. Wearable and Multimodal Path

### 26.1 Research Wearable Concept

The long-term hardware concept is an adaptive longitudinal recorder, not a generic hormone meter and not an epilepsy alarm. Candidate sensors:

- minimal scalp or forehead EEG leads selected from frontier-retention studies;
- ECG or PPG for cardiac context;
- IMU for movement and posture;
- optional EOG/EMG for sleep research;
- on-device signal-quality flags;
- event-triggered preservation of higher-resolution data.

### 26.2 Hardware Gate

Hardware work begins only when retrospective ablations identify a minimal montage with reproducible state/transition skill. The prototype must first prove measurement agreement and forecast calibration against simultaneous research-grade acquisition.

### 26.3 Gut-Brain, Endocrine, and Nexus Concepts

Gut-brain, inflammatory, endocrine, cortisol, alpha-amylase, and other body-context signals are not real-time neural ground truth and are not the Phase 0 product. They may later enter as time-aligned context variables only if:

- acquisition timing and units are validated;
- a causal availability time is defined;
- incremental value over EEG and autonomic baselines is tested;
- missingness and confounding are modeled;
- no biomarker is treated as a direct measure of dopamine, anhedonia, or central state.

---

## 27. Publication and Communication Protocol

### 27.1 Paper 1: Ruler and Phase 0

Possible title class:

> **A leakage-audited and externally calibrated framework for measuring the noninvasive neural predictability frontier**

Required paper structure:

1. unmet measurement need;
2. causal target and external split contract;
3. chief baselines and named adversaries;
4. state and transition frontiers;
5. calibration and negative controls;
6. external and reciprocal replication;
7. null/limited outcomes reported honestly;
8. claim and clinical limitations.

### 27.2 Figures

- causal context-gap-target timeline;
- recording-to-manifest-to-anchor pipeline;
- subject and external split diagram;
- baseline ladder;
- state horizon curve with confidence and calibration;
- transition competing-risk diagram;
- transition lead-time frontier;
- negative-control panel;
- calibration near/far from transitions;
- evidence gate and claim boundary.

No figure may show unsupported modalities as evaluated outputs. Proposed future modalities must be visually separated and labeled not evaluated.

### 27.3 Reporting Language

Use:

- "forecasted the declared future sensor/state target";
- "incremental skill over the frozen baseline";
- "held-subject" and "held-dataset" where true;
- "operational sleep macrostate transition";
- "calibrated under the declared external dataset";
- "non-clinical retrospective evidence."

Avoid:

- "understands the brain";
- "predicts seizures";
- "implant-grade";
- "digital twin";
- "brain foundation model";
- "clinical-ready";
- "diagnoses" or "treats";
- "first" without a formal systematic novelty review.

---

## 28. Requirements and Acceptance Criteria

### R1: Physical Recording Integrity

**Requirement:** Every admitted recording has validated physical and provenance metadata.

**Acceptance criteria:**

- [ ] Sampling rate, units, leads, reference, subject, session, and source are present.
- [ ] Missing or ambiguous metadata causes explicit exclusion or masking.
- [ ] Source and transform checksum status is recorded.

### R2: Split Integrity

**Requirement:** No held-out subject, session, recording, site, or dataset unit enters fitting.

**Acceptance criteria:**

- [ ] Splits are generated before transformations and anchors.
- [ ] Duplicate and same-person-night audits pass.
- [ ] Dataset and site claims match actual metadata.

### R3: Temporal Firebreak

**Requirement:** Forecast contexts and targets are strictly causal and disjoint.

**Acceptance criteria:**

- [ ] Context/target/filter-guard intersections are empty.
- [ ] Scored test anchors have no shared raw support.
- [ ] Injected overlap is detected automatically.

### R4: Baseline Sufficiency

**Requirement:** Kahlus is evaluated against strong causal statistical and neural comparators.

**Acceptance criteria:**

- [ ] Linear dynamics and semi-Markov chief baselines are present.
- [ ] Budgets and inputs are matched.
- [ ] Baseline choice is frozen on validation only.

### R5: Probabilistic Validity

**Requirement:** Forecast uncertainty is trained and evaluated.

**Acceptance criteria:**

- [ ] Proper score, calibration, and sharpness are reported.
- [ ] Calibration uses validation subjects only.
- [ ] Near-transition calibration is reported separately.

### R6: Negative Controls

**Requirement:** Known leakage and nuisance shortcuts are actively falsified.

**Acceptance criteria:**

- [ ] Time-shift, target permutation, unrelated-context, and identity controls run.
- [ ] Any positive forbidden control invalidates the run.
- [ ] Failed controls appear in the final report.

### R7: External Generalization

**Requirement:** Primary claims use an untouched external dataset.

**Acceptance criteria:**

- [ ] External manifest is sealed before model selection.
- [ ] No test-time retuning or recalibration occurs.
- [ ] Reciprocal direction is reported as replication.

### R8: Subject-Level Inference

**Requirement:** Population uncertainty reflects people, not windows.

**Acceptance criteria:**

- [ ] Subject is the outer bootstrap unit.
- [ ] Multiple horizons receive simultaneous inference.
- [ ] Seed variance is reported separately.

### R9: Distributed Correctness

**Requirement:** Multi-GPU execution preserves one coherent optimization trajectory.

**Acceptance criteria:**

- [ ] Every rank reports the same final optimizer step.
- [ ] Lockstep, collective, resume, and finite checks pass.
- [ ] World size and rank artifacts agree.

### R10: Evidence Reproducibility

**Requirement:** Every reported number traces to a clean commit and checksummed artifact.

**Acceptance criteria:**

- [ ] Full evidence bundle passes audit.
- [ ] Independent scoring reproduces primary metrics.
- [ ] Figures trace to source tables.

### R11: Claim Hygiene

**Requirement:** Conclusions remain within the evidence boundary.

**Acceptance criteria:**

- [ ] Allowed and blocked claims are machine-readable.
- [ ] No diagnosis, warning, treatment, invasive-parity, or causal claim appears without its required stage.
- [ ] Final gate and manuscript language agree.

### R12: Data Retention and Privacy

**Requirement:** Data remain available for reproducibility without entering prohibited artifacts.

**Acceptance criteria:**

- [ ] Raw/prepared shared caches are protected from cleanup.
- [ ] Raw public/private neural data are absent from git and evidence zips.
- [ ] Access and license rules are documented.

---

## 29. Definition of Program Completion

The initial program is complete when all of the following exist:

- frozen protocol and preregistration;
- trusted Sleep-EDF and CAP adapters;
- physical recording and lead registry;
- subject and external split manifests;
- strictly causal state and transition targets;
- full chief-baseline ladder;
- minimal probabilistic Kahlus model;
- calibration and subject-level inference;
- adversarial controls;
- sealed external and reciprocal results;
- machine-readable outcome class;
- independent evidence reproduction;
- publication-ready tables and figures;
- explicit next-stage or stop decision.

Completion does not require that Kahlus wins. It requires that the question has been answered under the frozen protocol.

---

## 30. Immediate Next Actions

1. Review this protocol with an EEG/sleep expert and a statistician.
2. Resolve the current branch divergence and incorporate the DDP lockstep fix without losing user work.
3. Freeze Sprint H0 thresholds, ontology, and primary external endpoint.
4. Implement contracts and audits before model code.
5. Build Sleep-EDF and CAP physical-data cards.
6. Run and publish the complete baseline-only horizon study.
7. Decide whether the signal supports a neural model before using the full A100 cluster.

---

## 31. Reference and Evidence Anchors

### Neuroscience and Measurement

- Buzsaki G, Anastassiou CA, Koch C. [The origin of extracellular fields and currents - EEG, ECoG, LFP and spikes](https://pubmed.ncbi.nlm.nih.gov/22595786/). Nature Reviews Neuroscience, 2012.
- Adamantidis AR, Gutierrez Herrera C, Gent TC. [Oscillating circuitries in the sleeping brain](https://pubmed.ncbi.nlm.nih.gov/31616106/). Nature Reviews Neuroscience, 2019.
- Saper CB et al. [Sleep state switching](https://pubmed.ncbi.nlm.nih.gov/21172606/). Neuron, 2010.
- de Zambotti M et al. [Dynamic coupling between the central and autonomic nervous systems during sleep](https://pubmed.ncbi.nlm.nih.gov/29608990/). Neuroscience and Biobehavioral Reviews, 2018.
- Lee YJ et al. [Interrater reliability of sleep stage scoring: a meta-analysis](https://pubmed.ncbi.nlm.nih.gov/34310277/). Journal of Clinical Sleep Medicine, 2022.

### Datasets

- [Sleep-EDF Database Expanded](https://physionet.org/content/sleep-edfx/1.0.0/). PhysioNet.
- [CAP Sleep Database](https://physionet.org/content/capslpdb/1.0.0/). PhysioNet.
- [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/). PhysioNet.
- [Siena Scalp EEG Database](https://physionet.org/content/siena-scalp-eeg/1.0.0/). PhysioNet.
- [EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/). PhysioNet.

### Current Landscape and Claim Boundaries

- Kweon YS et al. [SSF-SET: A Discrete EEG Token-based Framework for Sleep Stage Forecasting](https://pubmed.ncbi.nlm.nih.gov/41662557/). IEEE Journal of Biomedical and Health Informatics, 2026.
- Shafiezadeh S et al. [A systematic review of cross-patient approaches for EEG epileptic seizure prediction](https://pubmed.ncbi.nlm.nih.gov/39580818/). Journal of Neural Engineering, 2024.
- Kuruppu G et al. [EEG foundation models: a critical review of current progress and future directions](https://pubmed.ncbi.nlm.nih.gov/41666566/). Journal of Neural Engineering, 2026.
- Wang S et al. [A geometry aware framework enhances noninvasive mapping of whole human brain dynamics](https://www.nature.com/articles/s41551-026-01664-0). Nature Biomedical Engineering, 2026.
- Subramanian AK et al. [Scalp EEG predicts intracranial brain activity in humans](https://pubmed.ncbi.nlm.nih.gov/40291696/). bioRxiv preprint, 2025.
- Mikulan E et al. [Simultaneous human intracerebral stimulation and HD-EEG, ground truth for source localization methods](https://pubmed.ncbi.nlm.nih.gov/32345974/). Scientific Data, 2020.

### Forecasting and Machine Learning

- Gu A, Dao T. [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://openreview.net/forum?id=AL1fq05o7H). ICLR, 2024.
- Gneiting T, Raftery AE. Strictly Proper Scoring Rules, Prediction, and Estimation. Journal of the American Statistical Association, 2007.
- Scheuerer M, Hamill TM. [Variogram-Based Proper Scoring Rules for Probabilistic Forecasts of Multivariate Quantities](https://journals.ametsoc.org/view/journals/mwre/143/4/mwr-d-14-00269.1.xml). Monthly Weather Review, 2015.
- Xu C, Xie Y. [Conformal Prediction for Time Series](https://pubmed.ncbi.nlm.nih.gov/37819805/). IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023.

---

## 32. Final Governing Principle

> **Kahlus does not earn credibility by becoming larger or more complex. It earns credibility by defining a consequential question that can fail, constructing an evaluator that catches its own shortcuts, and reporting the result exactly as the evidence permits.**
