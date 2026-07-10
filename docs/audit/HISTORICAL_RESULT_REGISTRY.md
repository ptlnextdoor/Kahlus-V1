# Historical Forecasting Result Registry

## Ineligible Historical Protocol

Any result created with `kahlus.forecast.v1_overlap` is historical engineering evidence only. Its input and target windows may overlap, so it is not eligible for paper-mode future-window forecasting, baseline-superiority, biological, clinical, foundation-model, or Neural Field Compiler claims.

The recovered BNCI2014-001 artifact reporting MSE 3.116 is included in this category. It used a GRU-based `NeuralStateSpaceTranslator` path and must be described only as an overlapping-target recovered result.

## Registry Rules

- Preserve historical artifacts and checksums unchanged.
- Attach `forecast_protocol_id: kahlus.forecast.v1_overlap` when an artifact is loaded or summarized.
- Do not compare its metrics numerically with `kahlus.forecast.v2_nonoverlap` results.
- Any result without a verified protocol identifier is `protocol_unknown` and is ineligible for claim-bearing tables.
