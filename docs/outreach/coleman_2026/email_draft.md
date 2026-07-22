# Email draft — Coleman/Rao lab (DO NOT SEND)

**To:** Professor Todd Coleman (cc: Rao lab as appropriate)  
**Subject:** Residual forecastability engine for simultaneous EEG+EGG — follow-up to gut–brain sleep preprint

---

Dear Professor Coleman,

Thank you again for the SIMR talk and for the conversation afterward. I have been following your lab’s simultaneous stomach–brain electrophysiology work, including Rao et al. (bioRxiv 2025, *Simultaneous stomach-brain electrophysiology reveals dynamic coupling in human sleep*). What stood out is not only the dataset but the **evidence discipline**: nuisance covariates in LMMs, cluster-based permutation tests, circular-shift surrogates, and nested models with cross-validated ΔR² — including an honestly reported null (ΔR² = 0.0005, *p* = 0.84; ΔCVR² = −0.015).

I have been building **Kahlus**, a leakage-audited benchmark harness (Neural-CASP) for asking a closely related question in held-out subjects:

> After base rate, cycles, recent history, and EEG nuisance structure are removed, does a peripheral/autonomic block **Z** add residual forecastability for future state transitions?

We score that increment as **RFS bits** = (NLL_B − NLL_{B+Z}) / ln 2 — a cross-fitted, subject-cluster-bootstrapped cousin of your nested ΔCVR² test, with mandatory negative controls (label shuffle, time-shift, circular-shift surrogate, subject probe).

**Why I am writing now**

1. **Method fit.** Your Segment B nested-model logic and our RFS gate are aimed at the same scientific object: incremental value of gut/peripheral signal beyond brain + nuisance. Kahlus adds held-out-subject cross-fitting, explicit baseline ladders (including moving average / persistence), and artifact contracts (`claim_scope`, `stop_reason`) so positive and null results are equally publishable.

2. **Proof of competence (public proxy, not your data).** We ran an **interoception RFS scout** on public Sleep-EDF cassette PSG (EOG + EMG + respiration as **Z**, EEG + history as **B**, future sleep-stage transition as **Y**). This is explicitly **not** gastric EGG and **not** a claim about your cohort. The scout machinery passes synthetic known-signal / known-null fixtures; on real Sleep-EDF smoke the gate **failed honestly** (residual RFS did not survive controls at scout thresholds). That negative is exactly the kind of result the harness is designed to emit without over-claiming.

3. **Parallel honesty on EEG forecasting.** Separately, we audited isolated (non-overlapping) EEG forecasting on Sleep-EDF and BNCI2014_001. After removing the 126/127 input–target overlap illusion, **GRU does not beat the best trivial baseline at h=1** (BNCI: persistence 0.614 vs GRU 0.568; bootstrap gap CI excludes zero in the wrong direction). We are **not** leading with forecasting-skill claims.

**Ask**

Would you be open to a short call about applying the RFS / Neural-CASP gate to your simultaneous EEG+EGG data? We would deliver a scoped artifact bundle (JSON + Markdown): per-horizon RFS bits with cluster CIs, nested Δ-bits, cluster-permutation nulls, and pre-registered controls — with **no** clinical, consciousness, or “gut controls the brain” framing.

I am happy to share the one-pager (`docs/outreach/coleman_2026/one_pager.md`) and the public Sleep-EDF scout report before any data transfer.

Thank you for considering this.

Best,  
[Your name]

---

**Claim boundaries (internal — do not soften in outreach)**

- Not a clinical product, seizure predictor, or consciousness detector  
- Not claiming replication of Rao et al. on public Sleep-EDF  
- Not claiming EEG forecasting skill after isolated evaluation  
- Scope = **residual forecastability methodology** on held-out subjects
