# Claim-Evidence Matrix

The machine-readable source of this table is `claim_evidence_matrix.json`. Status labels describe scientific support, not code quality.

| ID | Claim | Existing evidence | Status | Permitted wording |
|---|---|---|---|---|
| C001 | Kahlus forecasts a distinct future EEG window | MSE 3.116 on a target sharing 126/127 samples with context | CONTRADICTED_BY_EVIDENCE | "Predicts a one-sample-shifted overlapping EEG sequence in a recovered run" |
| C002 | The recovered EEG evaluation holds out subjects | BNCI2014-001 split manifests list disjoint train/validation/test subjects | IMPLEMENTED_BUT_NOT_VALIDATED | "Used a subject-separated split in the recovered artifact" |
| C003 | Kahlus beats ridge on the recovered task | 3.116 versus ridge 7.745, but unequal optimization budgets and overlapping targets | IMPLEMENTED_BUT_NOT_VALIDATED | "Lower MSE than the repository ridge implementation on this recovered overlapping-target task" |
| C004 | The published result evaluates the Neural Field Compiler | Result config names `NeuralStateSpaceTranslator` with GRU fallback | CONTRADICTED_BY_EVIDENCE | Do not attribute 3.116 to NFC |
| C005 | NFC is a continuous neural field or neural operator | No coordinate input, resolution-transfer test, or function-space operator evidence | UNSUPPORTED | "Experimental latent-tensor dynamics architecture" |
| C006 | Real missing-modality reconstruction works | Recovered masked task MSE 53.977 and near-zero correlation; broad claim gate failed | CONTRADICTED_BY_EVIDENCE | "Masked reconstruction failed in the recovered evaluation" |
| C007 | Multimodal neural translation is validated | Paired-modality paths are synthetic or infrastructure-only | SYNTHETIC_ONLY | "Synthetic translation machinery exists" |
| C008 | Few-shot subject adaptation is validated | adaptation code and synthetic/local tests; no robust external cohort evidence | SYNTHETIC_ONLY | "Adaptation protocol is implemented for controlled fixtures" |
| C009 | Sleep transition forecastability is established | M2 uses six tiny Sleep-EDF pairs and labels itself machinery-only | IMPLEMENTED_BUT_NOT_VALIDATED | "Tiny public-data feasibility check" |
| C010 | Seizure forecastability is established | M3 is underpowered, gate fails, and nuisance accuracy is 1.0 | CONTRADICTED_BY_EVIDENCE | "Underpowered negative feasibility result" |
| C011 | ResearchDock provides reproducible evidence contracts | extensive tests and generated bundles; independent raw-to-result reproduction absent | INFRASTRUCTURE_ONLY | "Evidence-contract and audit infrastructure" |
| C012 | Uncertainty is calibrated | uncertainty head exists; recovered calibration artifact says unavailable | UNSUPPORTED | "Produces an uncalibrated uncertainty score" |
| C013 | Kahlus generalizes across sites/datasets/montages | adapters and configs exist; no completed external validation | UNSUPPORTED | "Cross-dataset evaluation is planned" |
| C014 | Neurovisual symptom maps are clinically validated | no clinical validation cohort | DOCUMENTATION_ONLY | "Research schema/prototype only" |
| C015 | Counterfactual or perturbation response is learned | synthetic transition/EM work only | SYNTHETIC_ONLY | "Synthetic counterfactual testbed" |
| C016 | Distributed scaling is validated | 3-GPU finalization artifact exists; a 7-GPU run failed from rank drift | IMPLEMENTED_BUT_NOT_VALIDATED | "Distributed runner has partial engineering evidence" |
| C017 | The system is a clinical digital twin or foundation model | no evidence supporting either designation | UNSUPPORTED | Block these terms |

## Manuscript Rule

No abstract, figure, caption, table, or conclusion may use wording stronger than the permitted wording above. Passing an artifact-contract gate is not equivalent to passing a scientific-validity gate.
