# Kahlus Forecastability Trial 0 - M5 Evidence Report

Gate passed: `True`
Synthetic gate passed: `True`
Gate failures: `none`
Claim scope: `passive_predictive_complexity_synthetic_method_only`
Validation scope: `synthetic_instrument_validity_only`
External generalization: `False`
Public data used: `False`
Nuisance conditioned: `True`
Bootstrap mode: `smoke`

## Method

Passive Predictive Integration Complexity estimates whether nuisance-conditioned joint future prediction beats nuisance-conditioned factorized channel-group future prediction under patient-held-out cross-fitting. It is a synthetic method gate only.
PIC bits are an operational predictive-integration score under this model class, not a direct measure of consciousness or integrated information.

| world | PIC bits | PIC CI low | PIC CI high | integration-feature residual RFS | RFS CI low | RFS CI high | gated baseline |
|---|---:|---:|---:|---:|---:|---:|---|
| integrated_predictive | 1.780134 | 1.602485 | 1.945034 | 0.043601 | 0.033534 | 0.055021 | logistic_nuisance |
| independent_predictable | -0.029815 | -0.045654 | -0.012349 | 0.001164 | -0.001879 | 0.004629 | logistic_nuisance |
| white_noise | -0.011366 | -0.027128 | 0.004227 | -0.001957 | -0.003841 | 0.000004 | logistic_nuisance |
| nuisance_only | -0.022511 | -0.033270 | -0.012670 | -0.001181 | -0.002804 | 0.000638 | logistic_nuisance |

## Attribution

Time-summary PIC is the primary gated score. Spectral-power PIC is reported for attribution only and is not required to be positive.

| world | time PIC bits | spectral PIC bits | spectral CI low | spectral CI high |
|---|---:|---:|---:|---:|
| integrated_predictive | 1.780134 | 0.000859 | -0.010763 | 0.015417 |
| independent_predictable | -0.029815 | 0.009588 | -0.001678 | 0.023905 |
| white_noise | -0.011366 | -0.003521 | -0.009832 | 0.001894 |
| nuisance_only | -0.022511 | 0.002572 | -0.003965 | 0.007913 |

## Blocked Claims

- `no_consciousness_claim`
- `no_pci_replacement_claim`
- `no_clinical_claim`
- `no_model_superiority_claim`

M5 is synthetic-only; public-data Passive PIC requires a later gate.
