# Kahlus-STF Benchmark Math Stop Note

This note is the required math stop before building Kahlus-STF models. It defines
the object, baselines, leakage modes, and falsification criteria for the passive
epilepsy/sleep monitoring lane.

## Modeled Object

Kahlus-STF models a latent transition state inferred from allowed history:

```text
h_t = {EEG history, optional body/sleep context, signal quality, missingness, allowed metadata}
z_t = f_theta(h_t)
```

The research target is not diagnosis. The target is whether `z_t` improves:

- future EEG forecasting
- longer-horizon EEG forecasting
- held-out channel or sensor completion
- calibrated event-risk windows when valid event labels exist

## Required Baselines

Baselines are results, not scaffolding. No STF model is interpretable until it
beats the appropriate baseline ladder.

| Task | Required baselines |
| --- | --- |
| future EEG forecasting | persistence, ridge/AR, TinySSM |
| longer-horizon EEG forecasting | persistence, ridge/AR, TinySSM |
| held-out channel completion | channel mean, ridge/AR, TinySSM |
| patient-held-out event-risk forecasting | cycle/time-of-day, event frequency, logistic ridge |

## Required Negative Controls

- shuffled-target control
- time-shifted-label control

Both controls must be split-safe. Training perturbations must not leak validation
or test labels, and event labels must not be shifted in a way that accidentally
uses post-event information.

## Leakage Modes That Could Fake Success

- adjacent-window autocorrelation
- overlapping windows across splits
- subject or patient identity leakage
- session/device/site leakage
- duplicated or near-duplicated segments
- post-event labels leaking into pre-event forecasts
- calibration tuned on the held-out test period
- event-frequency or time-of-day cycles mistaken for neural forecasting

## Falsification Criteria

STF stays a benchmark lane, not a model claim, if any of these happen:

- persistence, ridge/AR, TinySSM, or cycle baselines dominate
- shuffled or time-shifted controls stay close to real-task performance
- patient-held-out performance collapses relative to within-patient splits
- time-held-out calibration fails
- the model improves MSE but not clinically relevant review metrics
- event-risk claims cannot be evaluated from valid labels

## Claim Boundary

Allowed: passive benchmark, future signal forecasting, held-out sensor
completion, calibrated event-risk research windows, clinician-reviewed evidence.

Blocked: epilepsy diagnosis, seizure prevention, treatment, medication changes,
stimulation, replacing vEEG/PSG, or wearable-device efficacy.
