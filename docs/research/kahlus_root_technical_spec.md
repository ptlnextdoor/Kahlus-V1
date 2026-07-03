# Kahlus Root Technical Spec

## Scope

Kahlus is a staged research program for noninvasive brain/body state measurement. The root model should stay concrete:

```text
sensor/task history
  -> modality tokenizers
  -> temporal latent-state model
  -> perturbation-response operator
  -> future response profile + uncertainty
```

This is a research scaffold, not a clinical decision system.

## Versions

### Kahlus v1

Purpose: EEG forecasting and representation baseline.

```text
EEG patch tokenizer
+ SSM/Transformer encoder
+ latent z_t
+ future EEG decoder
```

Loss:

```text
L_v1 = lambda_1 MSE(Y_hat, Y) + lambda_2 MAE(Y_hat, Y) + lambda_3 L_smooth
L_smooth = sum_t ||Y_hat_{t+1} - 2Y_hat_t + Y_hat_{t-1}||_2^2
```

Role: strict baseline/evidence lane. Ridge and persistence baselines are results, not scaffolding.

### Kahlus v2

Purpose: multimodal observation-operator model.

```text
z_{t+1} = f_theta(z_t, u_t, s)
y_hat_t^(m) = O_m(z_t)
```

Initial operators:

```text
O_EEG = temporal decoder
O_pupil = MLP/readout
O_HRV = MLP/readout
O_behavior = MLP/readout
```

### Kahlus v3

Purpose: perturbation-response modeling.

```text
C_K(h_t) = [P(tau | h_t, a_1), ..., P(tau | h_t, a_K)]
```

Public datasets only partially support this. Full v3 needs structured history, task/perturbation, and future multimodal response data.

## Inputs And Outputs

Inputs may include EEG, webcam pupil/gaze, PPG/HRV, reaction time, task events, self-report sliders, context logs, and optional fNIRS later.

Outputs include future signal prediction, reward/stress response profile, uncertainty, and feature/quality reports.

## Baselines

Minimum baselines:

- persistence
- train mean
- linear ridge
- autoregressive ridge
- MLP
- TCN
- Transformer
- SSM fallback
- direct translation baseline for cross-modal tasks

Baselines come before Kahlus model claims.

## Evaluation Criteria

- subject-held-out split audit
- site/dataset-held-out audit when datasets support it
- no overlapping windows across splits
- finite metrics
- baseline table present
- quality flags reported
- evidence gate blocks broad and clinical claims

## Claim Boundaries

Allowed: response-profile measurement, reward/stress task prototype, biomedical research scaffold, non-diagnostic biomarker exploration, synthetic and prototype-stage evidence.

Blocked: diagnosis, treatment, depression/PTSD/anhedonia detection, epilepsy diagnosis, clinical decision system, proven brain foundation model, consciousness device, God Helmet, neurostimulation therapy.

## Citations To Verify

| Reference | Why It Matters | citation_status |
| --- | --- | --- |
| LFADS | neural population dynamics baseline | needs_verification |
| NDT | neural data Transformer baseline | needs_verification |
| BIOT | biosignal representation baseline | needs_verification |
| LaBraM | EEG representation baseline | needs_verification |
| NeuralBench | benchmark context | needs_verification |
| EEG Foundation Challenge | EEG benchmark context | needs_verification |
| WESAD | stress/physiology dataset candidate | needs_verification |
| DEAP | affect/physiology dataset candidate | needs_verification |
| SEED | affect EEG dataset candidate | needs_verification |
| CHB-MIT | seizure EEG dataset candidate | needs_verification |
| TUH/TUSZ | clinical EEG dataset candidate | needs_verification |
