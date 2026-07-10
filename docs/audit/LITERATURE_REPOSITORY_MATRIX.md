# Literature and External Repository Matrix

Consensus was used for discovery. Important statements were checked against primary proceedings, journal pages, official repositories, and source code fetched with `opensrc`. External code was not copied.

| System | Paper / venue | License | Exact source inspected with `opensrc` | Relevant practice | Kahlus action |
|---|---|---|---|---|---|
| MNE-Python | established neurophysiology software | BSD-3-Clause | `mne/io/base.py`, `mne/channels/montage.py`, `mne/io/edf/edf.py` | typed acquisition metadata, units, montages, canonical readers | reuse APIs; record exact MNE version and transforms |
| Braindecode | deep-learning toolbox for EEG | BSD-3-Clause | `braindecode/preprocessing/windowers.py`, `braindecode/datasets/bids/datasets.py`, `braindecode/models/eegnet.py` | window metadata and established EEGNet/ConvNet baselines | use tested baseline implementations where task-compatible |
| MOABB | Jayaram and Barachant, JNE 2018, DOI 10.1088/1741-2552/aadea0 | BSD-3-Clause | `moabb/evaluations/splitters.py`, `evaluations/base.py`, `evaluations/evaluations.py`, dataset/paradigm modules | reproducible dataset/paradigm/evaluation contracts | snapshot paradigm config and source version; do not call adapter use validation |
| LaBraM | ICLR 2024 spotlight, [OpenReview](https://openreview.net/pdf?id=QzTpTRVtrP) | MIT | `modeling_pretrain.py`, `modeling_vqnsp.py`, `norm_ema_quantizer.py`, `engine_for_pretraining.py` | channel patches, masked discrete-code pretraining, broad downstream evaluation | external pretrained baseline only after task compatibility and contamination audit |
| BIOT | NeurIPS 2023, [proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f6b30f3e2dd9cb53bbf2024402d02295-Abstract-Conference.html) | MIT | `model/biot.py`, `datasets/CHB-MIT/process1.py`, `datasets/CHB-MIT/process2.py`, `run_supervised_pretrain.py` | channel-wise spectral tokenization across mismatched datasets | compare as heterogeneous-biosignal baseline; reproduce preprocessing exactly |
| EEG Conformer | IEEE TNSRE 2023, DOI 10.1109/TNSRE.2022.3230250 | GPL-3.0 | `conformer.py`, `conformer_BCIIV2b.py`, `visualization/CAT.py` | compact local-plus-global EEG classifier | conceptual/executable external comparison only; do not copy GPL source |
| SIREN | NeurIPS 2020, [proceedings](https://proceedings.neurips.cc/paper_files/paper/2020/hash/53c04118df112c13a8c34b38343b9c10-Abstract.html) | MIT | `modules.py`, `dataio.py`, `loss_functions.py` | explicit coordinate-conditioned continuous signal representation | use only in a coordinate-to-signal reconstruction benchmark |
| NeuralOperator | JMLR 2023, [paper](https://www.jmlr.org/papers/v24/21-1524.html) | MIT | `neuralop/layers/spectral_convolution.py`, `neuralop/layers/fno_block.py`, `neuralop/models/tests/test_fno.py` | maps between function spaces; discretization invariance is central | require montage/resolution-transfer tests before using operator language |
| torchcde / Neural CDE | NeurIPS 2020, [proceedings](https://proceedings.neurips.cc/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html) | Apache-2.0 | `torchcde/interpolation_linear.py`, `interpolation_cubic.py`, `solver.py`, `example/irregular_data.py` | principled irregular and partially observed time series | relevant for asynchronous multimodal data, not current regular EEG task |
| brainmagick | Nature Machine Intelligence 2023, DOI 10.1038/s42256-023-00714-5 | CC BY-NC 4.0 | `bm/studies/api.py`, `bm/studies/download.py`, `bm/solver.py`, `bm/test_model.py` | multi-dataset/participant organization, spatial sensor attention, ablations | study patterns; do not copy into unrestricted code because license is noncommercial |
| Neural Latents Benchmark | NeurIPS Datasets and Benchmarks 2021, [paper](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/979d472a84804b9f647bc185a877a8b5-Paper-round2.pdf) | MIT tools | `nlb_tools/make_tensors.py`, `nlb_tools/evaluation.py`, `tests/test_evaluate.py` | fixed benchmark contracts and centralized evaluation | emulate immutable task/evaluator separation, with attribution |

## Source-Level Comparison

LaBraM and BIOT solve broad representation problems using explicit tokenization and large heterogeneous corpora. Their published classification metrics are not directly comparable to Kahlus forecasting MSE. EEG Conformer is also a classification architecture, not a future-signal baseline without adaptation and a new protocol.

SIREN defines a field as a neural function of coordinates. NeuralOperator defines an operator as a discretization-invariant map between function spaces. The current NFC accepts finite tensors, creates nodes from a source-to-target linear projection, and does not test resolution invariance. Those papers therefore provide falsification criteria, not automatic support for the NFC name.

MOABB, MNE, and Braindecode offer the highest immediate value: acquisition metadata, established preprocessing, subject-aware evaluation, and classical/deep EEG baselines. Kahlus should extend those contracts only where its forecasting and evidence-gating questions differ.

The exact Kahlus comparison points are `src/neurotwin/adapters/moabb.py`, `src/neurotwin/data/prepared_tasks.py`, `src/neurotwin/eeg_v1/dataset.py`, `src/neurotwin/models/torch_models.py`, `src/neurotwin/models/nfc/compiler.py`, and `src/neurotwin/benchmarks/baseline_suite.py`.

## Licensing Boundaries

- BSD, MIT, and Apache source can inform implementation subject to attribution and notice requirements.
- EEG-Conformer's GPL source must not be copied into a differently licensed Kahlus implementation without a deliberate licensing decision.
- brainmagick's CC BY-NC source is unsuitable for unrestricted reuse; use it as a methodological reference.
- Model weights and datasets may have terms distinct from repository code licenses and require separate review.
