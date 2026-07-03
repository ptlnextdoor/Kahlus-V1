# Kahlus Agent Constitution

> **Agents: read the [TL;DR](#tldr--read-this-first) first. This file wins over all other instructions unless the human explicitly overrides in the current message.**

Last updated: 2026-07-03 · Post PR #30 / Forecastability Trial 0 · [README](README.md) · [Conductor brief](CONDUCTOR.md)

---

## TL;DR — read this first

**What Kahlus is (one sentence):**

> Leakage-controlled research engine testing whether noninvasive biosignals contain **residual** information about future brain/body state transitions.

**What Kahlus is NOT:** seizure predictor · consciousness detector · brain foundation model · clinical product · the 3.116 MSE headline.

**Four layers — do not conflate:**

| # | Layer | One line |
| ---: | --- | --- |
| 1 | **Kahlus V1 (code)** | Neural-CASP courtroom: M0–M3 gates, manifests, RFS, artifacts |
| 2 | **DML-RPSM** | Defendant model; metric = **RFS bits** = \((\mathrm{NLL}_B - \mathrm{NLL}_{BZ}) / \ln 2\) |
| 3 | **Core question** | Does signal \(Z\) beat nuisance \(B\) on held-out subjects/sites/datasets? |
| 4 | **Moonshots** | Feature hypotheses plugged into the engine — not "Kahlus itself" |

**AlphaFold shape:** `Neural-CASP (arena)` × `Passive PCI (flagship defendant)` × `cross-etiology generalization`.

**Research priority (unless human redirects):**

```text
1. Harden Neural-CASP (moving-average baseline gate, artifact freshness, CI)
2. Track F1 — Passive PCI (passive EEG/MEG measure vs PCI / LZ / spectral baselines)
3. Track A — Criticality residual (ΔRFS_crit)
4. Track B — BCI↔GCI / EGG scout (only if public simultaneous data exists)
5. Tracks C–E — complexity index · nuisance-invariant audit · cross-modal law (long-term)
```

**Gate predicate (every scientific claim):**

```text
GatePass = split_disjoint ∧ finite ∧ beat_best_baseline ∧ controls_pass ∧ powered ∧ scoped_claim
```

**Baseline ladder (must beat the best, not just logistic-on-B):**

```text
moving average / persistence → ridge → GBM → standard EEG + B → incumbent (PCI, LZ, spectral)
```

**Controls (must collapse):** label shuffle · time-shift · cluster-permutation null · subject probe ≤ chance+0.2 · synthetic NULL must FAIL.

**Supported claim:**

> Leakage-aware benchmark harness for neural translation, EEG forecasting, transition forecastability, held-out reconstruction, adaptation, and evidence-gated reporting under public-data smoke.

**Never claim:** clinical seizure prediction · diagnosis · validated wearable · brain foundation model · A100 = biomedical utility · **3.116 MSE / 0.972 r as whole-model** (narrow forecast sidecar only; reconstruction failed; aggregate 28.55 MSE / 0.48 r).

**Artifact pattern (mandatory):** `runner → gate → JSON + Markdown` · committed artifacts must match code · raw neural data **never** in git.

**Before finishing:** read this file → map task to layer/track → `pytest -q` + `ruff check .` → state claim boundary in commit/PR.

**Mental model:** Kahlus = lie detector for neural forecasting claims. Gamble on hypotheses; **never** gamble on evidence standards.

→ Full detail below. Conductor workspaces: see also [CONDUCTOR.md](CONDUCTOR.md).

---

## Table of contents

1. [Four layers](#1-the-four-layers)
2. [AlphaFold program & Passive PCI](#2-the-alphafold-shaped-program)
3. [Claims & A100 metrics](#3-supported-vs-unsupported-claims)
4. [Evidence gates](#4-evidence-gates)
5. [Information atlas](#5-information-atlas)
6. [Research ladder & tracks](#6-staged-research-ladder)
7. [Moonshot deep dives](#7-moonshot-deep-dives)
8. [Core experiments](#8-core-experiments)
9. [Code conventions](#9-code-and-repo-conventions)
10. [Agent checklist](#10-agent-operating-checklist)
11. [Anti-patterns](#11-anti-patterns)
12. [Graphify & NeuroTwin v1](#12-graphify--neurotwin-v1-legacy-scope)
13. [References](#13-references)

---

## 1. The four layers

### Layer 1 — Kahlus V1 (Neural-CASP courtroom)

PR #30 scaffold:

```text
forecasting harness          M0–M4 gates
baseline tables              persistence, ridge, GBM, MLP, TCN, transformer, model
split manifests + leakage guards
negative controls            shuffle, time-shift, cluster bootstrap
RFS scoring                  residual forecastability score (bits)
public-data smoke            Sleep-EDF, CHB-MIT, TUSZ adapter
A100 evidence hygiene        aggregate ≠ task sidecars
claim gates                  JSON + Markdown artifacts
```

| Gate | Status | Meaning |
| --- | --- | --- |
| M0 | Passed | Evaluator determinism + baseline harness |
| M1 | Passed | Synthetic RFS (known signal + null) |
| M2 | Passed | Sleep machinery + held-out channel reconstruction (tiny subset) |
| M3 | Failed honestly | Underpowered CHB-MIT; TUSZ external not in committed evidence |
| M4+ | In progress | Passive PCI / DML-RPSM hardening |

**Agent rule:** Extend gates and harnesses. Do not bypass manifests, controls, or artifact contracts.

### Layer 2 — DML-RPSM V0

**Double-Machine-Learning Residual Predictive State Model:**

> After base rate, cycles, history, and subject/site nuisance are removed, does the signal block add anything?

```text
RFS_bits = (NLL_baseline - NLL_model) / ln(2)
         ≈ I(Y; Z | B)   (bits of conditional mutual information)
```

Implementation: `src/neurotwin/forecastability/m1.py` — `_crossfit_*`, `_rfs_bits`, `_cluster_bootstrap_rfs`. New estimators must use cross-fitting and the same leakage discipline.

### Layer 3 — Canonical research question

> **In held-out subjects, does a noninvasive sensor stream contain residual predictive information about future neural/body state transitions beyond temporal continuity, base rates, cycles, recent-history baselines, and subject/site identity?**

```text
Input:   X_{≤t}  (EEG / EGG / multimodal windows)
Target:  Y       (transition in horizon h)
Nuisance B: base rate + cycles + history + site/subject structure
Question:  does Z improve prediction beyond B?
```

### Layer 4 — Moonshot tracks (defendants)

| Track | Hypothesis | Priority |
| --- | --- | --- |
| **0 Neural-CASP** | Blind leakage-proof benchmark | Always build |
| **F1 Passive PCI** | Passive measure ≈ PCI discriminability | **Flagship** |
| **A Criticality** | Shared loss-of-resilience signature | Secondary |
| **B BCI↔GCI** | EGG adds info beyond EEG+autonomic | Dark horse |
| **C Complexity** | State-predictive complexity index | Secondary |
| **D Nuisance SSL** | Certified leak score / DML reps | Infrastructure |
| **E Cross-modal** | Universal latent across modalities | Long-term |

---

## 2. The AlphaFold-shaped program

AlphaFold won because **CASP existed** and the method **beat the incumbent on a blind benchmark**.

```text
Neural-CASP (Layer 1)   = arena
Passive PCI (Track F1)  = flagship defendant
Cross-etiology test     = universality (sleep → anesthesia → seizure under dataset-held-out)
```

### Passive PCI (Passive Consciousness-State Clock)

> Resting/spontaneous EEG/MEG measure that predicts held-out consciousness-relevant state labels and matches or beats **PCI** under blind cross-subject / cross-dataset evaluation — **without TMS hardware**.

- **Allowed:** "predictive complexity measure of neural state"
- **Forbidden:** consciousness solved · soul detected · clinical diagnosis
- **EM track:** Reanalyze open TMS-EEG/tES data only. **Never** design human stimulation hardware (God Helmet, productivity zappers, dosing).

PCI (Casali/Massimini) = TMS perturbation + LZ complexity of response. Passive PCI democratizes the **discriminability** problem onto cheap EEG.

---

## 3. Supported vs unsupported claims

### Supported

> Leakage-aware benchmark harness for neural translation, EEG future-window forecasting, state-transition forecastability, held-out channel reconstruction, subject adaptation, and evidence-gated reporting under public-data smoke settings.

### Unsupported — never in code, docs, commits, or marketing

- Clinical seizure prediction or epilepsy diagnosis
- Validated wearable or clinical decision system
- Brain foundation model / first multimodal brain model
- A100 completion proves biomedical utility
- 3.116 MSE / 0.972 Pearson as whole-model result
- Consciousness solved / metaphysical claims
- Consumer neuro-wearable before residual signal proven

### A100 metric hygiene (`6621642-ddpfix`)

| Scope | MSE | Pearson r | Note |
| --- | ---: | ---: | --- |
| Aggregate multitask | 28.546603 | 0.479801 | Mean of sidecars; never headline alone |
| `future_state_forecasting` | 3.116075 | 0.972108 | Narrow win; r² ≈ 0.945 |
| `masked_neural_reconstruction` | 53.977132 | -0.012507 | **Failed** |

Ledger: `docs/results/a100-run-history.md`

---

## 4. Evidence gates

```text
GatePass = C_split ∧ C_finite ∧ C_baseline ∧ C_controls ∧ C_power ∧ C_claim_scope
```

| Condition | Requirement |
| --- | --- |
| **C_split** | \(G_{\mathrm{train}} \cap G_{\mathrm{test}} = \emptyset\) on subject/site/dataset/modality key; audited manifests |
| **C_finite** | No NaN/Inf in payloads |
| **C_baseline** | Beat **best** trivial predictor (incl. moving average — CHB-MIT smoke: MA NLL 0.021 vs model 0.62) |
| **C_controls** | Shuffle + time-shift + permutation null collapse; probes ≤ chance+0.2; synthetic NULL fails |
| **C_power** | ≥8 event patients; ≥100 positive windows; ≥10–12 clusters for CI; n_boot ≥2000 |
| **C_claim_scope** | `claim_scope` + `stop_reason` in every artifact |

**Pattern:** `runner → gate → JSON + Markdown` · regenerate artifacts when gate logic changes.

---

## 5. Information atlas

Reporting frame (chain rule bookkeeping — **not** a novel theorem):

```text
baseline forecastability         I(Y; B)
subject structure                I(Y; S | B)     [report; don't leak]
residual sensor forecastability  I(Y; X | B, S)  [RFS]
cross-domain forecastability     I(Y; D | X, B, S) [dataset-held-out]
```

**If positive:** some features survive decomposition across domains (conserved loss-of-resilience).

**If negative:** most claimed neural forecasts are autocorrelation + subject identity + leakage.

Both are field-changing on Neural-CASP.

---

## 6. Staged research ladder

```text
1. Harden Neural-CASP (M0–M4, DML-RPSM, moving-average gate, artifact tests, CI)
2. Track F1 — Passive PCI benchmark + measure
3. Track A — ΔRFS_crit on Sleep-EDF + powered CHB-MIT/TUSZ
4. Track B — EGG scout (public data only)
5. Track C — Predictive complexity (de-consciousnessed)
6. Track D — Certified leak score
7. Track E — Cross-modal law (held-out modality B)
```

**Do:** use courthouse for stranger defendants · fail honestly · keep raw data out.

**Don't:** bigger EEG model as main bet · seizure wearable narrative · generic fusion without held-out splits · new framework when M-gates suffice.

---

## 7. Moonshot deep dives

### Track 0 — Neural-CASP

Field's missing blind benchmark for neural forecasting. EEG foundation models are fragmented; wrong splits inflate scores (Pup et al. 2025). PR #30 is the start. **Needs one decisive positive or negative result** to matter like CASP.

### Track F1 — Passive PCI (flagship)

Beat PCI / LZ / spectral / permutation entropy on held-out wake–NREM–REM–anesthesia labels. Cross-etiology: train sleep, test anesthesia/seizure with dataset held out. Highest WOW if it holds: cheap EEG replaces TMS lab for state assessment.

### Track A — Criticality residual

Hypothesis: approaching transitions share loss-of-resilience (autocorrelation, variance, 1/f slope, entropy, distance-to-criticality).

```text
ΔRFS_crit = RFS(B + EEG + criticality) - RFS(B + EEG)
```

Evidence: Maturana et al. intracranial critical slowing (Nature Commun. 2020). Caution: Dablander — EWS not model-agnostic; Helmich — weak clinical forecast evidence; scalp EEG ceiling is lower than iEEG.

### Track B — BCI↔GCI (interoceptive channel)

Hypothesis: \(I(Y; \mathrm{EGG} \mid \mathrm{EEG}, \mathrm{HRV}, \mathrm{Resp}, B) > 0\).

```text
ΔRFS_gut = RFS(B + EEG + EGG) - RFS(B + EEG)
```

Novel if true. Data bottleneck: simultaneous EEG+EGG public sets are small/preprint. Condition on respiration/HRV/meal timing or signal collapses.

### Track C — Predictive complexity

Compare LZ, permutation entropy, multiscale entropy, metastability, spectral slope, criticality distance on held-out state labels. **Never** brand as consciousness solved.

### Track D — Nuisance-invariant SSL

Target \(I(Z; N \mid Y) \le \epsilon\) with high \(I(Z; Y)\). DML cross-fitting is one route. Probe-at-chance ≠ proof. Use **Certified Leak Score** as audit, not primary science metric.

### Track E — Cross-modal translation law

\(X^{(m)} = g_m(Z_t, \eta)\). Attractive AlphaFold shape but **underidentified** (invertible transforms of \(Z\)). Long-term umbrella only. Real bar: modality A → held-out modality B on unseen subjects, beating all baselines.

---

## 8. Core experiments

### Experiment 1 — Criticality residual

Features: lag-1 autocorr, variance, CV, 1/f slope, multiscale entropy, Hurst/DFA, change-point proximity.

Ablations: `B` · `B+EEG` · `B+criticality` · `B+EEG+criticality` · shuffled/time-shifted criticality.

Win: \(\Delta RFS_{crit} > 0\), CI excludes 0, controls collapse, ≥2 domains, held-out subject.

### Experiment 2 — BCI↔GCI scout (data-gated)

Require: simultaneous EEG+EGG, state labels, ideally ECG/respiration, public access.

Controls: phase-shifted EGG · subject-shuffled EGG · HRV/resp-conditioned baseline.

Claim: "EGG carries residual state information" — not "gut controls consciousness."

### Experiment 3 — Passive PCI / complexity benchmark

Datasets: Sleep-EDF · open propofol/GABA sedation · Cam-CAN MEG · open TMS-EEG (PCI baseline).

Task: held-out subject state classification + ordinal depth RFS over spectral baseline.

Stretch: dataset-held-out · modality-held-out (EEG→MEG).

---

## 9. Code and repo conventions

### Package map

```text
src/neurotwin/forecastability/   M0–M4, RFS core
src/neurotwin/eeg_v1/            forecasting, adaptation
src/neurotwin/data/              schemas, split manifests
src/neurotwin/benchmarks/        baseline suite
src/stf/                         STF contracts (narrow)
scripts/run_forecastability_m*.py
artifacts/forecastability_trial0_*
docs/results/a100-run-history.*
tests/forecastability/
```

### Rules

- Minimize scope; match local conventions
- `PYTHONPATH=src python -m pytest -q` + `python -m ruff check .` before done
- Deterministic seeds; bit-stable tables (M0)
- No secrets; no raw neural data in git

### Known debt

- M3 committed artifact may be stale vs current `m3.py`
- RFS uses hand-rolled GD; prefer offset-GLM when refactoring
- `EEGV1SprintATests` hub oversized

---

## 10. Agent operating checklist

**Before:** read TL;DR → identify layer + track → check unsupported claims.

**During:** manifests not scans · runner→gate→artifacts · synthetic known+null · log blockers in `docs/research/`.

**After:** pytest + ruff · artifacts match code · claim boundary in commit · separate aggregate vs sidecar metrics.

---

## 11. Anti-patterns

| Anti-pattern | Why |
| --- | --- |
| Consciousness solved / clinical diagnosis | Claim hygiene |
| 3.116 MSE as whole model | Reconstruction failed |
| RFS without beating moving average | Autocorrelation exploit |
| Gate pass, no artifact | Not reproducible |
| Passes synthetic NULL | Bug |
| Subject probe ≫ chance | Leakage |
| Weakened controls | Fraudulent evidence |
| Raw clinical EEG in repo | Policy violation |
| Stimulation hardware design | Safety / IRB out of scope |
| "First foundation model" | False claim |

---

## 12. Graphify & NeuroTwin v1 (legacy scope)

### Graphify

- Read `graphify-out/GRAPH_REPORT.md` before architecture questions
- If `graphify-out/wiki/index.md` exists, navigate it
- After code changes: `graphify update .`

### NeuroTwin v1 (still valid, narrower than Kahlus Core)

- Leakage-proof Neural Translation: reconstruction, forecasting, few-shot adaptation under held-out splits
- NFC = Neural Field Compiler scaffold; Pair-Operator = baseline/ablation
- Competitors as baselines: TRIBE v2, BrainVista, **Brain-OF** (primary multimodal opponent), BrainOmni, Brain Harmony
- Brain-OF is the multimodal boss fight; do not claim empty territory
- Raw public neural data never committed

---

## 13. References

- Jumper et al. 2021 — AlphaFold / CASP14 (Proteins, Nature)
- Liu et al. 2026 — EEG Foundation Models: Benchmarking and Open Problems (arXiv)
- Pup/Zanola et al. 2025 — Data partitioning and EEG deep learning (Computers in Biology and Medicine)
- Maturana et al. 2020 — Critical slowing and seizure susceptibility (Nature Communications)
- Gervais & Boucher 2023 — Criticality and altered states scoping review (Front. Systems Neuroscience)
- Dablander et al. 2020 — Anticipating critical transitions (Psychological Methods)
- Helmich et al. 2024 — Early warning signals in psychopathology (Nature Reviews Psychology)
- Balasubramani et al. 2022 — Gut-brain electrophysiology and satiety (Sensors)
- Rao et al. 2025 — Stomach-brain coupling in sleep (bioRxiv preprint)
- Bach et al. 2021 — DoubleML (arXiv)
- Casali et al. 2013 — PCI / TMS-EEG consciousness measure

**Repo docs:** `docs/research/pr30_forecastability_branch_report.md` · `docs/results/a100-run-history.md` · `docs/research/neurotwin_project_state.md`

---

## 14. Human override

Explicit human override in the current message only. Default to this document.

---

*Kahlus = lie detector for neural forecasting claims. The courtroom is built. The flagship is Passive PCI. Prove it or kill it.*
