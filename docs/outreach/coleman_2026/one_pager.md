# Kahlus RFS × Coleman nested-model problem class

**Draft collaboration brief — DO NOT SEND without human review**

## The shared question

Rao, Coleman, et al. (bioRxiv 2025) test whether stomach electrical activity adds predictive value for brain dynamics during sleep using nested models and cross-validated ΔR² / ΔCVR², with rigorous nulls (including an honestly reported **ΔR² = 0.0005, p = 0.84**).

**Kahlus asks the same incremental-value question** with a leakage-audited engine:

```text
RFS_bits = (NLL_B - NLL_{B+Z}) / ln(2)  ≈  I(Y; Z | B)
```

| Component | Coleman stack | Kahlus Neural-CASP |
| --- | --- | --- |
| Outcome **Y** | Brain state / coupling metric | Future state transition (horizon *h*) |
| Nuisance **B** | EEG + covariates in LMM | EEG features + base rate + recent history |
| Increment **Z** | EGG / gastric features | Peripheral autonomic features |
| Increment test | Nested ΔR² / ΔCVR² | Cross-fitted RFS bits + cluster bootstrap CI |
| Nulls | Permutation, circular shift | Shuffle, time-shift, circular-shift surrogate, subject probe |
| Split discipline | Cohort + model CV | **Subject-held-out** GroupKFold + cross-fit |

Kahlus does **not** replace your mechanistic coupling analyses; it **tightens the incremental-information claim** under explicit leakage guards.

## What we would run on simultaneous EEG+EGG data

**Runner → gate → artifacts** (committed JSON + Markdown):

1. **Feature blocks** — handcrafted epoch features (or lab-standard features) for EEG (**B**) and EGG (**Z**)
2. **Horizon sweep** — transition labels at *h* ∈ {1, 2, 4, …} epochs
3. **RFS gate** — `_crossfit_residual_proba` (DML-style), cluster-bootstrap CIs, cluster permutation *p*
4. **Coleman-style stats layer** — nested B vs B+Z Δ-bits via GroupKFold, cluster-permutation *p*, subject-cluster bootstrap CIs
5. **Controls** — must collapse: label shuffle, time-shift, circular-shift surrogate on **Z**, subject-identity probe ≤ chance+0.2
6. **Scoped output** — `claim_scope` + `stop_reason` in every artifact

**Deliverable path (proposed):** `artifacts/coleman_egg_rfs/{scout_report.json, SCOUT_EVIDENCE_REPORT.md}`

## Public-data scout (proof of competence — not your data)

We implemented the same gate on **Sleep-EDF cassette** PSG as a public proxy:

- **Z:** EOG + EMG + respiration  
- **B:** EEG + history + base rate  
- **Y:** future sleep-stage transition  

**Result:** synthetic known/null fixtures **pass** (machinery validated). Real Sleep-EDF smoke: **gate failed honestly** — do **not** claim peripheral residual forecastability on that public proxy at scout thresholds.

Artifacts: `artifacts/interoception_rfs_scout/`

This is **scout-grade** evidence that the engine runs end-to-end; the decisive test is your simultaneous EEG+EGG cohort.

## Parallel result: EEG forecasting claim is dead

Isolated (strictly future-sample) forecasting audit:

| Dataset | h=1 best trivial | GRU | Verdict |
| --- | ---: | ---: | --- |
| Sleep-EDF (Amrith) | persistence 0.063 | 0.070 | GRU loses |
| BNCI2014_001 (C3, subj-held-out) | ridge 0.517 | 0.568 | GRU loses; gap CI [−0.086, −0.016] |

**We will not headline forecasting skill.** Kahlus contribution = courtroom + incremental-value rigor.

## Hard claim boundaries

**Supported**

- Leakage-aware incremental forecastability benchmark  
- Held-out-subject RFS bits with powered CIs and negative controls  
- Honest nulls as first-class results  

**Never claim**

- Clinical seizure prediction or diagnosis  
- Consciousness solved / PCI replacement  
- “Gut controls the brain”  
- 3.116 MSE / 0.972 *r* as whole-model performance  
- Sleep-EDF scout as replication of Rao et al.

## Suggested next step

30-minute call to align on:

1. Feature definitions for **B** (EEG) and **Z** (EGG) matching your nested models  
2. Subject / night split manifest  
3. Pre-registered horizons and control family  
4. Data-sharing / IRB constraints  

No data transfer implied by this document.
