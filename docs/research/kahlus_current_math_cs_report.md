# Kahlus Current Mathematics and Computer Science Report

Date: 2026-07-01

Scope: current `Kahlus-V1` worktree plus current 7xA100 handoff experiment at `/tmp/kahlus-7xa100-intensive-stf.tar.gz`.

Claim boundary: this codebase is a leakage-aware neural translation and benchmark framework. It is not a clinical diagnostic system, treatment system, seizure predictor, foundation model proof, or validated wearable.

## 1. Executive Summary

The current codebase implements a research harness around one core idea:

> infer or learn a compact latent state from partial noninvasive observations, then test whether that state helps predict future signals, reconstruct held-out channels or modalities, adapt to held-out subjects, and pass negative controls under leakage-aware splits.

The strongest implemented pieces are infrastructure:

- manifest-stage data records and split manifests
- windowing into supervised forecasting/reconstruction tasks
- brutal baselines: persistence, ridge, simple state-space/GRU, Transformer/MLP/TCN where used
- benchmark gates that block broad scientific or clinical claims
- EEG v1 future-window forecasting and few-shot subject adaptation smoke lanes
- STF epilepsy/sleep lane contract, synthetic smoke, CHB-MIT public-data smoke
- A100 packaging and audit infrastructure, plus a separate 7xA100 synthetic intensive stress runner

The most important mathematical fact is also the main scientific caution: short-horizon EEG future-window prediction is often dominated by autocorrelation. A low MSE does not imply biological understanding. That is why the code forces persistence, ridge, shuffled-target, time-shift, held-out-subject, and gate checks into the benchmark.

## 2. Repository-Level Architecture

Graphify reports 329 files, 3196 nodes, and 7964 edges. The main hubs are:

- `EEGV1SprintATests`
- `NumpyRidgeBaseline`
- data schemas, split manifests, leakage guards
- `TorchMLPBaseline`
- `TinyTransformerBaseline`
- `TinySSMBaseline`
- `SupervisedWindowTask`
- `DistributedInfo`
- `NeuralStateSpaceTranslator`

The codebase separates concerns reasonably cleanly:

```text
src/neurotwin/data/          schemas, records, split manifests, windows
src/neurotwin/scoring/       metrics and ranking
src/neurotwin/models/        baselines and neural translators
src/neurotwin/benchmarks/    task runners, baseline suite, reports
src/neurotwin/eeg_v1/        EEG v1 forecasting/adaptation benchmark lane
src/neurotwin/stf/           seizure-transition-forecasting benchmark lane
src/neurotwin/researchdock/  reward/anhedonia synthetic response-profile lane
src/neurotwin/transition_gym/ synthetic perturbation/operator-recovery lane
src/neurotwin/gates/         unified claim gate
src/neurotwin/training/      prepared-data training loop
src/neurotwin/a100_*         handoff/audit packaging
```

The repo is currently dirty and contains many untracked sprint files. That matters because the clean A100 handoff builder correctly refuses to package a final reproducible A100 handoff unless the worktree is clean.

## 3. Mathematical Data Model

Most tasks reduce neural observations to time-indexed tensors.

For a modality \(m\), subject \(s\), session \(r\), and time index \(t\):

\[
X^{(m)}_{s,r,t} \in \mathbb{R}^{C_m}
\]

where \(C_m\) is number of channels or features. For EEG, \(C_m\) is electrodes/channels. For ResearchDock, features include reward task variables, behavior, pupil, and HRV proxies.

A windowed sample is:

\[
\mathbf{X}_{i} = X_{t:t+L-1} \in \mathbb{R}^{L \times C}
\]

with target:

\[
\mathbf{Y}_{i}^{(h)} = X_{t+h:t+h+L-1}
\]

or, in STF shorter target form:

\[
\mathbf{Y}_{i}^{(h)} = X_{t+L:t+L+h-1}
\]

depending on task definition. The code has both idioms:

- `eeg_v1.dataset._future_windows()` uses equal input/target window length with forecast offset.
- `stf.smoke._forecast_arrays()` uses history window plus future horizon target.

The general supervised contract is `SupervisedWindowTask`:

```text
x_train, y_train, x_val, y_val, x_test, y_test
source_modality
target_modality
task_id
metadata
```

This is mathematically boring but important. It lets every baseline see the same tensors.

## 4. Splits and Leakage Control

The split system works at recording-manifest stage, before preprocessing/windowing. Each `RecordingRecord` has:

\[
(\text{record\_id}, \text{subject}, \text{session}, \text{site}, \text{dataset}, t_\text{start}, t_\text{end})
\]

The split policies are group splits or time splits:

- subject-held-out
- session-held-out
- site-held-out
- dataset-held-out
- chronological time split

For grouped splitting, the code partitions groups, not windows:

\[
G_\text{train} \cap G_\text{test} = \emptyset
\]

This blocks the common failure mode where adjacent windows from one subject/session leak into train and test. Each record also has a stable SHA-256 hash, so manifest identity can be audited.

Windowing then maps a recording into overlapping or strided windows:

\[
\{X_{0:L}, X_{\Delta:\Delta+L}, X_{2\Delta:2\Delta+L}, \dots\}
\]

Leakage caveat: overlapping windows are still allowed inside a split. That is fine for training efficiency, but not enough for scientific proof unless the split boundary is strict and negative controls fail as expected.

## 5. Metrics

The central regression metrics are:

Mean squared error:

\[
\mathrm{MSE}(Y,\hat{Y}) = \frac{1}{N}\sum_i (Y_i-\hat{Y}_i)^2
\]

Mean absolute error:

\[
\mathrm{MAE}(Y,\hat{Y}) = \frac{1}{N}\sum_i |Y_i-\hat{Y}_i|
\]

Coefficient of determination:

\[
R^2 = 1 - \frac{\sum_i (Y_i-\hat{Y}_i)^2}{\sum_i (Y_i-\bar{Y})^2}
\]

Pearson correlation:

\[
r = \frac{\mathrm{cov}(Y,\hat{Y})}{\sigma_Y\sigma_{\hat{Y}}}
\]

Rank-based and spectral checks also exist:

- Spearman rank correlation
- spectral FFT magnitude error
- bandpower error
- regionwise Pearson correlation
- bootstrap confidence intervals

The ranking rule is usually lowest MSE wins unless a metric is explicitly marked higher-is-better.

## 6. Baselines

### Persistence

Persistence predicts the future as the last observed state:

\[
\hat{X}_{t+1:t+h} = X_t
\]

For EEG, this is brutal because short-horizon EEG is smooth/autocorrelated. If Kahlus cannot beat persistence on a task, the task is mostly continuation.

### Ridge / Linear Autoregression

The ridge baseline flattens windows:

\[
x_i = \mathrm{vec}(\mathbf{X}_i), \quad y_i = \mathrm{vec}(\mathbf{Y}_i)
\]

Then solves:

\[
W^\* = \arg\min_W \|XW - Y\|_2^2 + \alpha \|W\|_2^2
\]

Closed form:

\[
W^\* = (X^\top X + \alpha I)^{-1}X^\top Y
\]

The implementation standardizes inputs and outputs with safe scaling, uses `np.linalg.solve`, and falls back to least squares if needed.

Why ridge can look strong:

- EEG is autocorrelated.
- Target is close in time to input.
- Normalization compresses error scale.
- Synthetic fixtures are clean.
- Linear continuation can capture much of the benchmark.

### TinySSM

There are two notions:

1. In `models/torch_models.py`, `TinySSMBaseline` is a GRU sequence model:

\[
h_t = \mathrm{GRU}(W_x x_t, h_{t-1}), \quad \hat{y}_t = W_o h_t
\]

2. In `stf.smoke`, `_tiny_ssm_forecast()` is a very small diagonal state-transition estimate:

\[
\hat{x}_{t+1,c} = a_c x_{t,c}
\]

where:

\[
a_c = \frac{\sum_i x_{i,c}y_{i,c}}{\sum_i x_{i,c}^2 + \epsilon}
\]

The second one is intentionally minimal: enough to test whether a learned/state-space-ish transition beats persistence without importing Mamba or heavy SSM libraries.

### Transformer

The tiny Transformer baseline maps:

\[
X \in \mathbb{R}^{B \times T \times C}
\]

through:

\[
H_0 = XW_\text{in}
\]

then self-attention layers:

\[
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
\]

and output projection:

\[
\hat{Y}=H_LW_\text{out}
\]

This is standard sequence modeling. It is not a novel architecture by itself.

### Negative Controls

Shuffled-target control:

\[
(x_i,y_i)\rightarrow(x_i,y_{\pi(i)})
\]

where \(\pi\) is a random permutation inside the training split. If shuffled targets score close to real targets, the task or metrics are suspect.

Time-shifted label control:

\[
y_t \rightarrow y_{t-k}
\]

Used in event-risk forecasting. If time-shifted labels perform similarly to true labels, the event-risk model is exploiting base rates/cycles rather than real predictive state.

## 7. Neural State Space Translator

The main neural abstraction is `NeuralStateSpaceTranslator`.

For each observed modality \(m\), the tokenizer maps native feature dimension into latent dimension:

\[
z_t^{(m)} = E_m(x_t^{(m)}) + e_m
\]

where:

- \(E_m\) is modality encoder.
- \(e_m\) is learned modality embedding.

If multiple modalities are present, fusion is a simple mean:

\[
z_t = \frac{1}{|\mathcal{M}|}\sum_{m\in\mathcal{M}} z_t^{(m)}
\]

Optional metadata, geometry, and subject embeddings are additive:

\[
z_t \leftarrow z_t + E_\text{meta}(u_t) + E_\text{geom}(g_t) + E_\text{subj}(s)
\]

The shared dynamics core is either a GRU/SSM fallback or Transformer-style backbone:

\[
h_{1:T} = F_\theta(z_{1:T})
\]

Heads map latent state to predictions:

\[
\hat{y}_t^{(m)} = O_m(h_t)
\]

There are separate heads for:

- reconstruction
- forecast
- readout
- projection embedding

The architecture is best described as a shared latent observation model. It is not currently a proven neural foundation model or digital twin.

## 8. Prepared Training Loop

The prepared-data training loop uses AdamW and MSE:

\[
\mathcal{L} = \frac{1}{N}\|\hat{Y}-Y\|_2^2
\]

with optional gradient accumulation:

\[
\nabla \mathcal{L}_\text{step} =
\frac{1}{K}\sum_{k=1}^K \nabla \mathcal{L}_k
\]

The loop includes:

- finite initial-loss check
- finite per-microbatch loss check
- optional gradient clipping
- validation snapshots
- best-checkpoint selection by `val_mse`
- final test evaluation using selected-best model
- DDP wrapping if distributed runtime is initialized

This is standard supervised sequence regression infrastructure.

## 9. EEG v1 Lane

EEG v1 creates synthetic or local EEG-like datasets with subject/session metadata. The synthetic fixture builds latent autoregressive states:

\[
\ell_t = 0.72\ell_{t-1} + \epsilon_t
\]

then projects them into EEG channel space:

\[
x_t = \ell_t P + \text{sinusoidal terms} + \text{subject/session effects}
\]

The forecasting task is:

\[
X_{t:t+L-1}\mapsto X_{t+h:t+h+L-1}
\]

under subject-held-out split. The few-shot adaptation task pretrains on train subjects, then for each held-out subject uses:

- support windows
- query windows
- persistence
- support ridge
- linear probe
- bottleneck adapter
- full finetune

The adaptation objective remains MSE. The key scientific question is not "does adapter train?" but:

> Does adaptation improve over support-only persistence/ridge under subject-held-out query windows?

The gate only allows:

```text
eeg_future_forecasting_benchmark_ready
eeg_fewshot_adaptation_benchmark_ready
```

It does not allow clinical or model-superiority claims.

## 10. STF Epilepsy / Sleep Lane

STF means seizure-transition forecasting in this codebase, but the current implementation is still a benchmark lane, not a clinical predictor.

Required STF tasks:

1. future EEG forecasting
2. longer-horizon EEG forecasting
3. held-out channel completion
4. patient-held-out event-risk forecasting

Required baselines:

```text
future / longer horizon: persistence, ridge_ar, tiny_ssm
held-out channel: channel_mean, ridge_ar, tiny_ssm
event risk: cycle_time_of_day, event_frequency, logistic_ridge
```

Required negative controls:

```text
shuffled_target_control
time_shifted_label_control
```

Required splits:

```text
patient_held_out
time_held_out
```

### Synthetic STF Fixture

The synthetic signal evolves:

\[
x_t = 0.82x_{t-1} + 0.10\sin(2\pi t/24 + \phi)c + \epsilon_t + b_s
\]

where:

- \(c\) is channel scale.
- \(b_s\) is patient shift.
- burst/event labels depend on periodic bursts and circadian thresholding.

This makes persistence and cycle baselines meaningful on purpose. The benchmark is designed to catch models that only learn smooth continuation or time-of-day risk.

### Event Risk

Event-risk rows use features:

\[
\phi_t = [\mathrm{mean}(x_{t-6:t}), x_t-x_{t-1}]
\]

and label:

\[
y_t \in \{0,1\}
\]

Baselines:

- event frequency:

\[
\hat{p} = \frac{1}{N}\sum_i y_i
\]

- cycle/time-of-day:

\[
\hat{p}(h)=\mathbb{E}[y\mid \text{hour}=h]
\]

- logistic ridge:

\[
\hat{p}=\sigma(w^\top x+b)
\]

with gradient updates and small \(L_2\) penalty.

Metric is Brier-like squared probability error:

\[
\mathrm{Brier} = \frac{1}{N}\sum_i (\hat{p}_i-y_i)^2
\]

### CHB-MIT Public Smoke

The CHB-MIT public-data smoke:

- audits that raw data is outside repo
- requires `RECORDS`, `RECORDS-WITH-SEIZURES`
- requires seizure annotation files for seizure records
- reads EDF files via `edfio`
- parses seizure start/end seconds from summary text
- converts times to samples using EDF sampling frequency
- selects patient-held-out/time-held-out windows

The smoke is intentionally small. It proves ingestion and benchmark plumbing, not clinical validity.

## 11. Current 7xA100 Intensive Experiment

Current archive:

```text
/tmp/kahlus-7xa100-intensive-stf.tar.gz
```

This package is a standalone compute-heavy synthetic STF runner. It is not yet integrated as a clean committed repo handoff.

Configuration:

```text
GPUs: exactly 7x A100 80GB
torchrun processes: 7
wall time: 12 hours
default target runtime: 12 hours
model: 20-layer Transformer
latent width: 1024
heads: 16
sequence length: 512
channels: 64
batch per GPU: 16
precision: bf16
compile: true
```

Mathematically, this experiment uses synthetic autoregressive/circadian EEG-like sequences:

\[
x_t = 0.96x_{t-1}+0.04\epsilon_t + 0.15\sin(3\omega t + r)c
\]

It trains a Transformer with three heads:

Future signal:

\[
\hat{x}_{T+1}=O_f(h_T)
\]

Masked reconstruction:

\[
\hat{x}_{t,\mathcal{M}}=O_m(h_t)_{\mathcal{M}}
\]

Event-like state:

\[
\hat{y}=\sigma(O_e(\mathrm{mean}_t h_t))
\]

Loss:

\[
\mathcal{L} =
\mathrm{MSE}(\hat{x}_{T+1},x_{T+1})
+ \mathrm{MSE}(\hat{x}_{\mathcal{M}},x_{\mathcal{M}})
+ 0.1\,\mathrm{BCEWithLogits}(\hat{y},y)
\]

Computer-science mechanics:

- `torchrun --standalone --nproc_per_node=7`
- NCCL backend when CUDA is available
- `DistributedDataParallel`
- per-rank deterministic seeds
- exact GPU count check
- exact A100 80GB name check
- rank-0 writes JSONL metrics and checkpoints
- target-hours deadline stops long run

This experiment is useful to test:

- DDP launch plumbing
- multi-GPU utilization
- memory pressure
- bf16/autocast path
- checkpoint writing
- long-running numerical stability

It is not useful as evidence for epilepsy, seizure prediction, clinical utility, or real-data model performance.

## 12. ResearchDock Lane

ResearchDock is a synthetic reward/anhedonia response-profile lane.

Task:

\[
x_i = [\text{reward condition}, \text{effort}, \text{reaction time}, \text{accuracy}, \text{self-report}, \dots]
\]

Target:

\[
y_i = [\text{pupil diameter}, \text{HRV proxy}]
\]

Models:

- train mean
- linear ridge
- observation operator

The observation operator is currently a lightweight structured model, not a validated biomedical device. The lane enforces subject-held-out split and missing-modality audit.

## 13. Transition Gym / v3 Lane

Transition Gym is synthetic operator-recovery machinery.

It creates a world with perturbation operators:

\[
x_{t+1}=A_k x_t + \epsilon
\]

for perturbation \(k\). Diagnostics test:

- operator recovery
- held-out composition recovery
- non-commutativity
- response-profile distances

Non-commutativity asks whether:

\[
A_iA_j \neq A_jA_i
\]

and whether the benchmark can expose that. KTM is included as an informational scaffold. The code explicitly says untrained KTM is expected to lose to ridge and that losing does not invalidate the synthetic benchmark.

## 14. KTM Training v3 Objective

The KTM objective combines trajectory error, profile error, and Gaussian NLL.

Trajectory MSE:

\[
\mathcal{L}_\text{traj}=\mathbb{E}\|\hat{X}_{1:H}-X_{1:H}\|^2
\]

Profile MSE:

\[
\mathcal{L}_\text{profile}=\mathbb{E}\|\bar{\hat{X}}-\bar{X}\|^2
\]

or full profile tensor error when available:

\[
\mathbb{E}\|\hat{C}_K - C_K\|^2
\]

Gaussian NLL with predicted log variance \(s=\log\sigma^2\):

\[
\mathcal{L}_\text{nll}=
\frac{1}{2}\mathbb{E}\left[s + \frac{\mathrm{SE}}{\exp(s)}\right]
\]

Total:

\[
\mathcal{L} =
w_\text{traj}\mathcal{L}_\text{traj}
+ w_\text{profile}\mathcal{L}_\text{profile}
+ w_\text{nll}\mathcal{L}_\text{nll}
\]

There is also a loss-explosion guard using a sliding median:

\[
\text{explode if } \mathcal{L}_t > \gamma \cdot \mathrm{median}(\mathcal{L}_{t-w:t-1})
\]

This is numerical safety infrastructure, not scientific validation.

## 15. Evidence Gates

The unified evidence gate is deliberately conservative:

Required booleans:

```text
split_audit_passed
baseline_table_present
finite_metrics
calibration_checked
```

Allowed claim scopes are narrow:

```text
eeg_future_forecasting_benchmark_ready
eeg_fewshot_adaptation_benchmark_ready
stf_epilepsy_benchmark_definition_ready
synthetic_transition_operator_recovery
researchdock_synthetic_response_profile
...
```

The gate blocks anything outside the allowlist. The STF gate also blocks terms like:

```text
diagnosis
diagnostic
treatment
medication
prevention
replaces_veeg
replaces_psg
clinical_predictor
stimulation
```

This is the correct design choice. Claims are treated as outputs of evidence, not marketing text attached to runs.

## 16. Computer-Science Design Choices

Good choices:

- dataclasses for task/config records
- simple NumPy/PyTorch boundary
- source-of-truth split manifests
- JSON/CSV artifacts for auditability
- deterministic seeds
- explicit finite checks
- no raw public EDF data committed
- no private raw participant data in packages
- package builders include checksums
- gates separate benchmark readiness from model superiority

Weaknesses:

- many lanes are synthetic or smoke-level
- worktree is dirty, so exact clean reproducibility is blocked
- STF real-data smoke is not a full benchmark
- A100 intensive runner is outside repo and synthetic-only
- current heavy A100 run has no resume support
- no real large-scale public-data DDP training path is ready
- some baselines are intentionally minimal stand-ins

## 17. Complexity and Scaling

Ridge fit:

\[
O(ND^2 + D^3)
\]

where \(N\) is number of samples and \(D\) flattened input dimension.

GRU/TinySSM:

\[
O(BTLd^2)
\]

roughly, for batch \(B\), time \(T\), layers \(L\), hidden dimension \(d\).

Transformer:

\[
O(BLT^2d + BLTd^2)
\]

The \(T^2\) term dominates for long windows. The 7xA100 run uses \(T=512\), \(d=1024\), \(L=20\), batch \(16\) per GPU, making it intentionally compute-heavy.

DDP communication cost is dominated by gradient synchronization:

\[
O(P)
\]

per step, where \(P\) is parameter count. Compute/communication ratio improves with larger per-GPU batch and model size.

## 18. Current Scientific Interpretation

Current supported statement:

> Kahlus has a working leakage-aware benchmark harness for EEG/STF-style forecasting, reconstruction, adaptation, baseline comparison, negative controls, and evidence-gated claim boundaries.

Current unsupported statements:

- Kahlus diagnoses epilepsy.
- Kahlus predicts seizures clinically.
- Kahlus replaces sleep studies, vEEG, or PSG.
- Kahlus is AlphaFold for neurology.
- Kahlus proves a brain foundation model.
- A100 synthetic loss reduction proves biomedical utility.

The code is pointed in the right direction: difficult baselines, stricter splits, and negative controls come before bigger models.

## 19. Verification Commands

Core checks:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/stf -v
PYTHONPATH=src python3 scripts/run_stf_synthetic_smoke.py --out-dir /tmp/kahlus_stf_smoke --seed 0
PYTHONPATH=src python3 scripts/run_stf_public_data_audit.py --dataset chb_mit_physionet --data-root /tmp/kahlus_chbmit_smoke_subset --out-dir /tmp/kahlus_stf_chbmit_audit
PYTHONPATH=src python3 scripts/run_stf_chb_mit_smoke.py --dataset chb_mit_physionet --data-root /tmp/kahlus_chbmit_smoke_subset --out-dir /tmp/kahlus_stf_chbmit_smoke --max-records 4 --max-samples-per-record 900000 --max-channels 8
```

7xA100 package smoke:

```bash
tar -xzf /tmp/kahlus-7xa100-intensive-stf.tar.gz
cd kahlus-7xa100-intensive-stf
python -m pip install -r requirements.txt
bash scripts/run_smoke.sh
```

7xA100 cluster run:

```bash
TARGET_HOURS=12 sbatch scripts/run_full.sbatch
```

Diff hygiene:

```bash
git diff --check -- ':!graphify-out'
```

## 20. Recommended Next Technical Step

Do not make the A100 run bigger until the real-data benchmark is cleaner.

Next best sprint:

1. Commit or isolate current STF work into a clean branch.
2. Build a clean 7xA100 handoff from exact commit.
3. Add real public-data STF training task, not synthetic-only stress.
4. Keep required baselines and negative controls in the same run.
5. Audit returned evidence before interpreting numbers.

That is the shortest path from "compute stress package" to "scientifically useful cluster run."
