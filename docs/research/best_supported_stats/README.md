# Kahlus best supported statistics

Generated from the local versions archive and current docs artifacts. This report is intentionally claim-hygienic: it highlights the strongest truthful results and names the places where baselines or gates block broader claims.

Source scan: `docs/research/best_supported_stats/best_supported_stats_scan.json`.

## Best honest headline

**Kahlus v1 has a narrow, paper-usable EEG future-forecasting win.** In the recovered 6621642 A100 handoff, Kahlus v1 achieved:

- **Task:** `future_state_forecasting`, EEG→EEG
- **MSE:** **3.11607456072082**
- **MAE:** **1.29685807808655**
- **Pearson r:** **0.9721078350075555**
- **R²:** **0.9418946133048665**
- **Evidence bundle:** `neurotwin-6621642-ddpfix-resume-handoff.zip`
- **Artifact:** `neurotwin-a100-results-6621642-evidence/run/tables/task_results.csv`

This beats the saved linear ridge baseline on the same EEG future-forecasting task:

- **Linear ridge MSE:** **7.7452734660906435**
- **Comparison:** Kahlus v1 recovered is about **2.49× lower MSE** than linear ridge for EEG future forecasting.

Safe wording:

> Kahlus v1 shows a recovered, batch-matched, leakage-audited EEG future-state forecasting result on MOABB/BNCI2014-style evidence, reaching MSE 3.116, Pearson r 0.972, and R² 0.942, outperforming linear ridge on the same future-forecasting metric.

## Best secondary EEG result

The best saved Kahlus-like EEG masked reconstruction row is strong in isolation but does **not** beat ridge:

- **Task:** `masked_neural_reconstruction`, EEG→EEG
- **Best Kahlus-family MSE:** **13.666385313467636**
- **Pearson r:** **0.864414321272169**
- **R²:** **0.745168376704751**
- **Evidence bundle:** `2026-06-02_neurotwin-a100-results-e9de1d2-paper-audit-v1-evidence.zip`
- **Artifact:** `neurotwin-a100-results-e9de1d2-evidence/run/tables/task_results.csv`
- **Linear ridge MSE:** **7.807342737229374**
- **Comparison:** ridge wins masked reconstruction.

Safe wording:

> Kahlus-family runs show nontrivial masked reconstruction signal, but the saved linear ridge baseline remains stronger on masked reconstruction. This blocks any full-suite superiority claim.

## Recovered checkpoint caveat

The recovered 6621642 checkpoint is the source of the best future-forecasting result, but it fails masked reconstruction:

- **Recovered future forecasting:** MSE **3.116075**, Pearson r **0.972108**, R² **0.941895**
- **Recovered masked reconstruction:** MSE **53.977132**, Pearson r **-0.012507**, R² **-0.006490**

Safe wording:

> The recovered checkpoint is excellent for future-state forecasting, but not a general EEG model: its masked reconstruction result fails badly.

## fMRI / Algonauts results

Saved fMRI rows have low MSE values because the target scale is different, but the correlations and R² are near zero. These are **not** strong model-success claims:

- `future_state_forecasting`, fMRI→fMRI: MSE **0.3704057158577742**, Pearson r **0.021147489604090967**, R² **0.00022823194044319095**
- `stimulus_to_fmri_response`, stimulus→fMRI: MSE **0.3700777378938426**, Pearson r **0.031562457078157746**, R² **0.0009851610724053161**
- Evidence bundle: `2026-06-01_neurotwin-algonauts2025-v1-evidence.zip`

Safe wording:

> Current fMRI/Algonauts rows are best treated as pipeline smoke/evidence artifacts, not as strong predictive performance.

## Gates and claim hygiene

The 6621642 evidence has mixed gate status:

- `paper_mode_gate.json`: **passed = true**, observed seeds `[0, 1, 2]`, no violations or warnings.
- `evidence_gate.json`: **passed = false**, `scientific_claim_allowed = false`.
- `evidence_gate_provisional.json`: **passed = false**, `scientific_claim_allowed = false`.

Interpretation:

- Paper-mode/reproducibility checks support discussing the narrow recovered forecasting result.
- The final evidence gate blocks broad claims like “Kahlus is better than all baselines,” “Kahlus is a general EEG foundation model,” or “Kahlus solves neural reconstruction.”

## Best marketing-safe claims

Use these:

1. **Best result:** Kahlus v1 beats linear ridge on EEG future-state forecasting: MSE **3.116** vs **7.745**, Pearson r **0.972**, R² **0.942**.
2. **Claim hygiene:** The same evidence system blocks unsupported full-suite claims because masked reconstruction fails or loses to ridge.
3. **Benchmark honesty:** Kahlus includes strong classical baselines, leakage audits, paper-mode gates, and evidence bundles, so wins and losses are both visible.
4. **Narrow technical promise:** Kahlus is promising for short-horizon EEG future forecasting, not yet proven as a general neural foundation model.

Do **not** say:

- “Kahlus beats ridge overall.” It does not.
- “Kahlus beats all models on masked reconstruction.” It does not.
- “Kahlus has proven clinical/diagnostic value.” It has not.
- “The fMRI results are strong.” Current fMRI correlations and R² are near zero.

## One-sentence version

Kahlus v1’s strongest honest result is a narrow but real recovered EEG future-forecasting win, with MSE **3.116** and R² **0.942**, beating linear ridge on that task, while the same evidence gates correctly block broader model-superiority claims because ridge still wins masked reconstruction and the recovered checkpoint fails that task.
