# NeuroTwin Technical Report

Date: 2026-05-24

## 1. Executive Summary

Yes: TRIBE v2 changes the plan. BrainVista and Brain-OF sharpen the change even further.

The original NeuroTwin plan was pointed in the right direction, but it underweighted how crowded the obvious lanes already are. Meta's TRIBE v2 is a tri-modal video/audio/language model that predicts high-resolution human fMRI responses across naturalistic and experimental conditions, trained on more than 1,000 hours of fMRI across 720 subjects, and explicitly framed as in-silico neuroscience. Its model card reports a CC-BY-NC-4.0 license, average-subject predictions on fsaverage5, and a Transformer pipeline using LLaMA 3.2, V-JEPA2, and Wav2Vec-BERT feature extractors.

BrainVista also takes a bite out of "future fMRI from stimulus": it models naturalistic brain dynamics as multimodal next-token prediction, uses history-only stimulus-to-brain masking, validates on Algonauts 2025/CineBrain/HAD, and reports better long-horizon rollout. Brain-OF directly crowds generic fMRI/EEG/MEG unification with an Any-Resolution Neural Signal Sampler, DINT attention, Sparse MoE, and masked temporal-frequency modeling across about 40 datasets. BrainOmni crowds EEG/MEG tokenization with BrainTokenizer and a sensor encoder for layout, orientation, and sensor type.

So NeuroTwin should not be framed as "TRIBE but more modalities," "first multimodal brain foundation model," or "first universal neural tokenizer." Those claims are too close, and likely lose.

The defensible breakthrough bet is:

**Leakage-proof Neural Translation: a rigorous benchmark plus subject-adaptive, multi-timescale architecture that maps among fMRI, EEG, MEG, anatomy, behavior, stimulus, and eventually spikes; reconstructs missing modalities; forecasts future latent brain states; and proves generalization under held-out-subject, held-out-site, and held-out-dataset splits.**

This is narrower and tougher. TRIBE v2 is a powerful stimulus-to-fMRI encoder. BrainVista is a strong naturalistic fMRI rollout model. Brain-OF is a strong fMRI/EEG/MEG unified foundation model. NeuroTwin should become the translation and evaluation layer those systems do not fully solve:

```text
stimulus + anatomy + subject context + observed neural modality
    -> shared latent neural state
    -> future state, missing modality, behavior, perturbation-conditioned counterfactual
```

The first paper should not claim a clinical digital twin. It should claim a new representation and evaluation framework:

**NeuroTwin: a leakage-proof neural translation benchmark and architecture for missing-modality reconstruction, future-state rollout, and few-shot subject adaptation across fMRI, EEG/MEG, anatomy, behavior, and stimulus-aligned data.**

## 2. Brutal Landscape Map

### fMRI and neuroimaging foundation models

This lane is crowded.

| Work | Solved | Failed to solve |
| --- | --- | --- |
| BrainLM | Transformer masked-prediction pretraining on 6,700 hours of fMRI; zero-shot/fine-tuning evidence. Source: https://openreview.net/forum?id=RwI7ZEfR27 | Mostly fMRI time-series abstraction. Not a multi-modal neural translator. Weak causal/intervention story. |
| Brain-JEPA | JEPA-style fMRI dynamics representation learning. Source: https://papers.nips.cc/paper_files/paper/2024/file/9c3828adf1500f5de3c56f6550dfe43c-Paper-Conference.pdf | Still fMRI-specific. Not a universal modality bridge. |
| NeuroSTORM | Mamba/shifted-window 4D fMRI foundation model, 28.65M fMRI frames, over 9,000 hours, over 50,000 subjects, multi-center ages 5-100. Source: https://arxiv.org/abs/2506.11167 and https://www.nature.com/articles/s41551-026-01666-y.pdf | Strong fMRI platform, but not EEG/MEG/spikes. Primarily analysis and downstream tasks, not intervention-conditioned dynamics. |
| Brain-DiT | Diffusion Transformer with metadata-conditioned pretraining on 349,898 sessions from 24 fMRI datasets across states. Source: https://arxiv.org/abs/2604.12683 | More general fMRI states, but still fMRI. Diffusion pretraining does not itself solve cross-modal temporal mismatch. |
| BrainHarmonix / Brain Harmony | Structural MRI + fMRI unified into compact 1D brain tokens; 64,594 T1 MRI volumes and 70,933 fMRI time series; handles heterogeneous TRs. Source: https://arxiv.org/abs/2509.24693 | Important token-compression precedent, but only structural MRI + fMRI. No EEG/MEG/spike translator. |
| BrainGFM | Brain graph foundation model using graph contrastive learning and masked graph autoencoding across atlases/disorders. Source: https://arxiv.org/abs/2506.02044 | Useful atlas/disorder graph formulation, but fMRI graph only. Clinical classification risk is high. |
| BrainSymphony | Parameter-efficient fMRI + structural-connectivity fusion with spatial/temporal Transformer streams, Perceiver compression, signed graph Transformer, adaptive fusion. Source: https://arxiv.org/abs/2506.18314 | Strong structural-function coupling, but limited cross-modal scope and not a universal state translator. |
| BrainVista | Naturalistic fMRI as multimodal autoregressive next-token prediction; Network-wise Tokenizers, Spatial Mixer Head, S2B masking, Algonauts/CineBrain/HAD validation, strong long-horizon rollout. Source: https://arxiv.org/abs/2602.04512 | Crowds future-fMRI rollout. Still fMRI-centered and stimulus/history driven, not a general neural-modality translation benchmark. |

Verdict: fMRI foundation modeling is now an arms race. A new fMRI-only model is not enough.

### EEG and MEG foundation models

This lane has scale, but the benchmark situation is messy.

| Work | Solved | Failed to solve |
| --- | --- | --- |
| BENDR | Early large-scale EEG self-supervision and transferable EEG features. Source: https://github.com/SPOClab-ca/BENDR | EEG-only, older scale, limited unified task coverage. |
| LaBraM / LaBraM++ | Large EEG foundation model; later codebook/signal-processing improvements. Source: https://arxiv.org/abs/2405.18765 and https://papers.cool/arxiv/2505.16724 | Strong EEG representation, but sensor layout, montage, low SNR, and dataset shift remain hard. |
| EEGPT | 10M-parameter pretrained Transformer with masked dual self-supervised learning and spatio-temporal representation alignment. Source: https://paperswithcode.com/paper/eegpt-pretrained-transformer-for-universal | Good universal EEG extractor, but EEG-only and not spatially rich. |
| NeuroLM | Treats EEG as a foreign language: VQ temporal-frequency neural tokenizer, causal autoregression, instruction tuning; about 25,000 EEG hours and up to 1.7B params. Source: https://arxiv.org/abs/2409.00101 | Creative EEG-language bridge, but not grounded in fMRI/anatomy or causal neural dynamics. |
| BrainOmni | Unified EEG + MEG foundation model with BrainTokenizer, sensor encoder for spatial layout/orientation/type, 1,997 EEG hours and 656 MEG hours, and unseen-device generalization. Source: https://arxiv.org/abs/2505.18185 and https://github.com/OpenTSLab/BrainOmni | Directly crowds EEG/MEG tokenization. Still electrophysiology-centered and does not solve fMRI/anatomy/stimulus translation. |
| Brain-OF | Joint fMRI, EEG, MEG model with Any-Resolution Neural Signal Sampler, DINT attention, sparse MoE, and masked temporal-frequency modeling across about 40 datasets. Source: https://arxiv.org/abs/2602.23410 | The new boss fight. It overlaps hard with generic multimodal neural foundation modeling. NeuroTwin must beat it on translation tasks, subject adaptation, leakage-proof held-out site/dataset tests, and anatomy/behavior/stimulus integration. |
| NeuroAtlas | Benchmark, not model: 42 EEG datasets, 260k hours, clinical EEG + BCI; finds EEG FMs are not consistently better than generic time-series FMs. Source: https://arxiv.org/abs/2605.14698 | This is a warning shot: EEG-only FM claims are weak unless evaluation is brutal. |
| NeuralBench | Meta benchmark framework: EEG/MEG/fMRI tasks, 36 EEG tasks, 94 EEG datasets, 9.5k+ subjects, 13.6k+ hours, 14 models. Source: https://github.com/facebookresearch/neuroai/blob/main/neuralbench-repo/README.md | Benchmark coverage is valuable; fMRI/MEG coverage still developing. |

Verdict: EEG/MEG should be NeuroTwin's fast temporal evidence stream, not the whole claim.

### Spikes, neural population, and BCI models

| Work | Solved | Failed to solve |
| --- | --- | --- |
| LFADS | RNN/VAE latent dynamics for spike trains, denoising and latent population dynamics. Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC6380887/ | Classical and strong, but not foundation-scale or multimodal across human neuroimaging. |
| NDT / NDT2 / NDT3 | Transformer foundation model for intracortical motor decoding; NDT3 pretrained on 2,000 hours from over 30 monkeys and humans from 10 labs. Source: https://openreview.net/forum?id=ONOe6cAE9I and https://pmc.ncbi.nlm.nih.gov/articles/PMC11838490/ | Excellent BCI/motor lane. Domain remains narrow relative to whole-brain human cognition. |
| Neuroformer | Multimodal, multitask GPT-style generative pretraining for systems neuroscience data. Source: https://arxiv.org/abs/2311.00136 and https://github.com/a-antoniades/Neuroformer | Good systems-neuro template, but limited data scale and not a human multimodal digital twin. |
| CEBRA | Contrastive joint behavioral-neural latent embeddings across species and modalities. Source: https://www.nature.com/articles/s41586-023-06031-6 | Great representation learner, but not a forecasting/reconstruction foundation model by itself. Patent/licensing notes matter. |
| OmniMouse | Large-scale mouse multimodal/multitask brain model with 150B neural tokens. Source: https://arxiv.org/abs/2604.18827 | Great mechanistic training source, but animal-to-human translation remains speculative. |
| IBL Brain-Wide Map | Large standardized Neuropixels decision-making resource. Source: https://www.internationalbrainlab.com/brainwide-map | Powerful for brain-region routing and behavior, not direct human twin. |

Verdict: spikes should teach mechanistic population dynamics and cross-session adaptation. Do not pretend mouse or motor cortex equals a human clinical brain twin.

### Digital twin and mechanistic simulation

| Work | Solved | Failed to solve |
| --- | --- | --- |
| The Virtual Brain | Whole-brain network simulation with connectome/neural-mass style dynamics. Source: https://www.thevirtualbrain.org/ | Mechanistic, but not foundation-scale learned from massive heterogeneous data. |
| Virtual Epileptic Patient / EPINOV | Personalized epilepsy spread modeling using MRI, EEG/sEEG, Bayesian inference; clinical-trial context. Sources: https://www.medrxiv.org/content/10.1101/2022.01.19.22269404v1 and https://pmc.ncbi.nlm.nih.gov/articles/PMC12043236/ | Disease-specific, hand-specified equations, not a general neural foundation model. |
| EU Neurotwin | Personalized hybrid brain models integrating electric fields, neural-mass physiology, and multimodal neuroimaging for stimulation effects. Source: https://cordis.europa.eu/project/id/101017716/reporting | Strong mechanistic/intervention framing. Not a large learned multimodal foundation model. |
| Human Brain Project / EBRAINS | Infrastructure and simulation services. Source: https://ebrains.eu/ | Powerful ecosystem, but no single learned universal translator. |
| Blue Brain Project | Detailed cellular/circuit simulations. Source: https://www.epfl.ch/research/domains/bluebrain/ | Very mechanistic, limited whole-human scalable data-driven generalization. |
| Dynamic Causal Modeling / neural mass models | Causal hypotheses and interpretable dynamic equations. | Usually small-scale, assumption-heavy, difficult to scale to foundation data. |
| CognitiveTwin | Robust multimodal digital twins for Alzheimer's cognitive decline. Source: https://arxiv.org/abs/2604.22428 | Disease-specific and high risk for overclaiming clinical utility. |

Verdict: classical digital twins have the intervention mechanism. Foundation models have the scale. NeuroTwin's opportunity is the bridge, but v1 must be research-grade, not clinical-grade.

### Multimodal stimulus-to-brain models

| Work | Solved | Failed to solve |
| --- | --- | --- |
| TRIBE | Tri-modal Brain Encoder for whole-brain fMRI response prediction; won Algonauts 2025. Source: https://arxiv.org/abs/2507.22229 and https://github.com/facebookresearch/algonauts-2025 | Naturalistic stimulus -> fMRI, not persistent neural-state world model. |
| TRIBE v2 | Video/audio/language foundation model for in-silico neuroscience, over 1,000 hours fMRI, 720 subjects, novel stimuli/tasks/subjects, recovers known neuroscience effects. Source: https://arxiv.org/abs/2605.04326 and https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/ | Still fMRI response prediction. No EEG/MEG/spikes/anatomy translation, no individual fast adaptation claim, no intervention-conditioned dynamics. |
| BrainVista | Multimodal autoregressive naturalistic fMRI prediction with prior brain states and current stimuli; strong rollout results. Source: https://arxiv.org/abs/2602.04512 | Makes "future fMRI from stimulus" insufficient as novelty. Does not own cross-modal neural translation. |
| Multimodal Seq2Seq Transformer | Predicts fMRI from visual, auditory, and language features plus prior brain states; shared encoder and partially subject-specific decoder. Source: https://arxiv.org/abs/2507.18104 | Confirms stimulus-to-fMRI is an active lane. Not a cross-modality neural-state translator. |
| fMRI-to-image/language/audio reconstruction | Strong decoding/reconstruction via CLIP/diffusion/LLMs, often NSD-centered. | Mostly reconstructs stimulus semantics from fMRI; not a causal or persistent brain-state model. |
| CineBrain/CineSync | Paired EEG/fMRI narrative stimulus data and audiovisual decoding ideas. Source: https://huggingface.co/papers/2503.06940 | Small subject count, but strategically valuable paired-modality alignment data. |

Verdict: TRIBE v2 and BrainVista make stimulus-to-fMRI a baseline, not the novelty. Brain-OF makes generic fMRI/EEG/MEG unification crowded. NeuroTwin must move from "foundation model" to "leakage-proof neural translation benchmark plus subject-adaptive translation architecture."

## 3. Dataset Table and Usable Value

| Dataset/source | Modalities | Scale | Access/licensing | Same-subject multimodal? | Best use |
| --- | --- | --- | --- | --- | --- |
| OpenNeuro | MRI, PET, MEG, EEG, iEEG, NIRS in BIDS. Source: https://openneuro.org/ | 1,500+ public BIDS datasets via BIDS ecosystem. | Open, dataset-specific licenses. | Sometimes. | Broad benchmark ingestion, BIDS normalization. |
| DANDI | NWB electrophysiology, optophysiology, behavior, associated images. Source: https://docs.dandiarchive.org/introduction/ | GitHub superdataset reports 1021 dandisets and 912.7 TB; article reports 400+ dandisets and 350+ TB at writing. Source: https://github.com/dandi | Public, dandiset-specific. | Sometimes. | Spike/calcium/behavior mechanistic training. |
| HCP Young Adult | MRI/fMRI/dMRI/behavior. Source: https://www.humanconnectome.org/study/hcp-young-adult/data-release/1200-subjects-data-release | 1206 healthy adults. | Public-domain-ish HCP access terms. | Yes, rich neuroimaging. | Clean structure-function anchor. |
| UK Biobank | Brain/body MRI, health, genetics, repeats. Source: https://www.ukbiobank.ac.uk/enable-your-research/about-our-data/imaging-data | 100,000 imaging participants; repeat imaging growing. | Application-controlled. | Yes across imaging/phenotype. | Population priors, anatomy/fMRI. |
| ABCD | Longitudinal brain/behavior children. Source: https://abcdstudy.org/about/ | About 11,880 children. | Controlled access. | Yes. | Developmental adaptation and longitudinal tests. |
| ADNI | MRI/PET/clinical/cognitive biomarkers. | Controlled. | Yes. | Secondary clinical transfer only. |
| TUH EEG Corpus | Clinical EEG. Source: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2016.00196/full | Large clinical corpus; NeuroAtlas includes clinical EEG at large scale. | Account/license required. | EEG + reports. | EEG robustness, artifact/pathology benchmarks. |
| NeuralBench/NeuralSet | EEG, MEG, fMRI benchmark/data loading. Source: https://github.com/facebookresearch/neuroai | 36 EEG tasks, 94 EEG datasets, 9.5k+ subjects, 13.6k+ hours. | MIT code; data licenses vary. | Mixed. | Standardized evaluation and loaders. |
| NSD | 7T fMRI natural images, 8 subjects, large repeated stimuli. Source: https://naturalscenesdataset.org/ | High-quality small-N visual fMRI. | Data-use terms. | fMRI + stimuli. | Stimulus-brain alignment and OOD. |
| Algonauts / CNeuroMod | Naturalistic video fMRI. Source: https://algonautsproject.com/2025/challenge.html | Strong stimulus-aligned fMRI. | Challenge/data terms. | fMRI + video/audio/text features. | TRIBE baseline and stimulus-response tests. |
| Allen Brain Observatory | Mouse visual cortex calcium/Neuropixels. Source: https://observatory.brain-map.org/visualcoding/ | Standardized mouse visual data. | Open. | Neural + stimuli/behavior. | Mechanistic visual cortex dynamics. |
| IBL Brain-Wide Map | Mouse Neuropixels decision-making. Source: https://www.internationalbrainlab.com/brainwide-map | Large multi-area mouse data. | Open. | Neural + behavior. | Region routing, behavior coupling. |
| MICrONS | Mouse visual cortex functional/connectomics. | Very large, complex. | Open/structured access. | Anatomy/connectivity + activity/stimuli. | Structure-to-function constraints. |
| Paired EEG-fMRI datasets | EEG + fMRI simultaneous or aligned. | Scarce and small. | Fragmented. | Yes, but limited. | Critical cross-modal translation validation. |

Blunt point: raw terabytes do not matter if event timing, subject IDs, preprocessing, site metadata, and stimulus labels are broken. The first engineering moat is metadata harmonization and leakage control.

## 4. Core Bottlenecks

1. **Temporal mismatch:** spikes/EEG/MEG live at milliseconds; fMRI lives at seconds; disease progression lives at months or years.
2. **Spatial mismatch:** spikes are local populations; EEG/MEG are inverse-problem sensors; fMRI is voxel/parcel/cortical surface; anatomy is structural.
3. **Subject identity leakage:** models can learn person/scanner/session fingerprints instead of brain function.
4. **Site/scanner leakage:** multi-site neuroimaging rewards confound exploitation unless splits are brutal.
5. **Metadata entropy:** task labels, stimulus timing, preprocessing histories, atlases, montages, and sensor layouts are inconsistent.
6. **Same-subject multimodal scarcity:** paired EEG-fMRI-spikes-anatomy data is rare. Translation requires clever weak-pairing and teacher-student objectives.
7. **Intervention scarcity:** clinical intervention/treatment data is too thin for broad causal claims.

## 5. Breakthrough Gap Candidates

| Candidate | Verdict |
| --- | --- |
| Universal Neural Tokenizer | Important but not empty territory. BrainOmni and Brain-OF already attack this. NeuroTwin's tokenizer is publishable only if tied to leakage-proof translation, uncertainty/provenance, and anatomy/stimulus/behavior alignment. |
| Cross-Modal Neural Translator | Strongest first-paper claim if framed as benchmark plus model. Missing-modality prediction is measurable and hard, and Brain-OF makes the comparison concrete. |
| Subject-Adaptive Neural State Space | High value. Needs few-shot adaptation benchmarks with 5-20 minutes of data. |
| Causal NeuroTwin | Highest long-term upside, but weak v1 evidence. Use as constrained objective, not headline claim. |
| Multi-Timescale Brain Dynamics Model | Real architectural gap. A hierarchical state-space model is more natural than a flat Transformer. |
| Brain-State World Model | Best long-term vision. Too broad for first paper unless scoped to forecasting + translation. |

Recommended synthesis:

**NeuroTwin v1 = leakage-proof neural translation benchmark + multi-timescale state-space translator + few-shot subject adaptation.**

## 6. Candidate Architectures

### Architecture 1: Neural Token Transformer

- Tokenization: modality-specific encoders produce geometry-aware tokens with time, brain region/sensor, subject, task, stimulus, and uncertainty embeddings.
- Backbone: Perceiver/Transformer with cross-attention over modality tokens.
- Objective: masked multimodal reconstruction, cross-modal retrieval, subject contrastive alignment.
- Adaptation: LoRA/adapters and subject embeddings.
- Strength: easiest to implement and compare.
- Failure mode: becomes a generic multimodal Transformer with no neural-specific contribution.
- Rank: high feasibility, medium novelty.

### Architecture 2: Neural State Space Translator

- Tokenization: event-based continuous-time tokens for fMRI/EEG/MEG/spikes/stimuli/anatomy.
- Backbone: hierarchical Mamba/SSM core with slow state for fMRI/anatomy, fast state for EEG/MEG/spikes, and cross-scale synchronization gates.
- Objective: future-state prediction, missing-modality reconstruction, stimulus-conditioned response prediction, contrastive alignment across subjects/tasks.
- Adaptation: amortized subject adapters plus test-time adaptation on 5-20 minutes of data.
- Strength: directly attacks temporal incompatibility and forecasting.
- Failure mode: optimization instability, hard negative transfer across modalities, and weak differentiation from Brain-OF unless the translation/adaptation benchmark is central.
- Rank: best first-paper architecture.

### Architecture 3: Causal NeuroTwin

- Tokenization: neural state tokens plus connectome/anatomy tokens plus perturbation/stimulation/lesion tokens.
- Backbone: foundation encoder plus differentiable neural-mass/DCM-inspired dynamics layer or connectome-constrained graph SSM.
- Objective: intervention-conditioned forecasting, counterfactual consistency, uncertainty calibration.
- Adaptation: Bayesian subject adapters and posterior parameter inference.
- Strength: highest long-term scientific distinctiveness.
- Failure mode: not enough intervention data; risk of fake causality.
- Rank: long-term flagship, not v1 headline.

Architecture ranking:

| Architecture | Novelty | Feasibility | Publishability | Risk | Breakthrough chance |
| --- | --- | --- | --- | --- | --- |
| Neural State Space Translator | High | Medium | High | Medium-high | Highest |
| Neural Token Transformer | Medium | High | Medium | Medium | Medium |
| Causal NeuroTwin | Highest | Low-medium | Medium if scoped | High | High long-term |

## 7. Recommended NeuroTwin Architecture

Build **NeuroTwin-Translator**, not just NeuroTwin.

Core modules:

1. **Universal Neural Tokenizer**
   - fMRI: surface/parcel/voxel tokens with TR, atlas, site, scanner, task, and region embeddings.
   - EEG/MEG: sensor-position tokens, montage graph, time-frequency patches, sampling-rate embeddings.
   - Spikes/calcium: population tokens with unit/region/session embeddings and bin-width metadata.
   - Anatomy/connectome: structural tokens and graph edges as slow constraints.
   - Stimulus/behavior: video/audio/text embeddings, event timings, behavior covariates.
   - Each token carries uncertainty/noise and provenance.

2. **Multi-Timescale State-Space Core**
   - Fast SSM stream: EEG/MEG/spikes.
   - Slow SSM stream: fMRI/anatomy/longitudinal state.
   - Cross-scale gates: learn when fast evidence updates slow latent state and when slow anatomy constrains fast predictions.
   - Transformer cross-attention only for expensive global fusion, not every timestep.

3. **Cross-Modal Translation Heads**
   - EEG/MEG -> fMRI-like latent.
   - fMRI -> EEG spectral envelope.
   - stimulus -> brain state.
   - brain state -> behavior.
   - spikes -> shared population/cognitive latent.
   - anatomy/connectome -> dynamic constraints.

4. **Subject Adaptation**
   - Global model frozen or semi-frozen.
   - Subject adapters from 5, 10, 20 minutes of calibration data.
   - Hypernetwork emits LoRA/adapters from subject metadata and initial recordings.
   - Uncertainty increases under unseen site/device/modalities.

5. **Mechanistic Constraint Layer**
   - Not full causal twin yet.
   - Add connectome smoothness, region-delay priors, DCM/neural-mass-inspired regularizers, and intervention tokens only where data supports them.

## 8. First Publishable Paper Idea

Title direction:

**Neural Translation: Leakage-Proof Cross-Modal Brain State Prediction and Few-Shot Subject Adaptation**

Main claim:

> NeuroTwin introduces a leakage-proof neural translation benchmark and a subject-adaptive state-space translator that improves missing-modality reconstruction and future latent brain-state forecasting across fMRI, EEG/MEG, anatomy, behavior, and stimulus-aligned datasets, outperforming TRIBE v2/BrainVista-style stimulus-to-fMRI baselines, Brain-OF-style fMRI/EEG/MEG unification, BrainOmni-style EEG/MEG tokenization, and modality-specialist baselines under held-out-subject, held-out-site, and held-out-dataset splits.

What it is not:

- Not a clinical diagnosis model.
- Not a perfect digital twin.
- Not "train giant model on all brain data."
- Not just TRIBE with extra modalities.
- Not the first brain foundation model.
- Not the first multimodal brain model.
- Not the first stimulus-to-brain model.
- Not the first neural tokenizer.

Minimum paper experiments:

1. **Benchmark harness first:** frozen split manifests, leakage reports, modality-held-out tasks, and subject/site/dataset-held-out evaluation.
2. **Stimulus + past fMRI -> future fMRI** vs TRIBE v2, BrainVista, seq2seq Transformer, linear, Mamba/SSM baselines.
3. **EEG/MEG -> shared latent brain state** vs Brain-OF, BrainOmni, EEGPT/LaBraM-style baselines.
4. **fMRI -> EEG/MEG spectral-state proxy** where paired data exists.
5. **Anatomy/fMRI -> subject-conditioned latent state** using HCP/UKB/ABCD-style data.
6. **Missing-modality reconstruction** across available modalities.
7. **Few-shot subject adaptation** with 5/10/20 minutes of calibration.
8. Spikes population forecasting vs LFADS/NDT/Neuroformer baselines as an extension, not the gating claim.
9. Cross-modal weak-pair translation:
   - paired EEG-fMRI where available,
   - pseudo-paired stimulus-conditioned alignment where same stimuli exist across modalities,
   - teacher-student distillation from TRIBE v2-like fMRI state into EEG/MEG fast state.

## 9. Ablation Plan

Mandatory:

- no geometry-aware tokenizer
- no uncertainty/provenance tokens
- no subject adapters
- no multi-timescale SSM, replace with flat Transformer
- no cross-modal missing-modality objective
- no future-state forecasting objective
- no anatomy/connectome constraint
- no leakage guard
- TRIBE/BrainVista-style stimulus-only path vs neural-state path
- Brain-OF-style shared multimodal backbone vs explicit translation heads
- BrainOmni-style electrophysiology tokenizer vs NeuroTwin provenance/uncertainty tokenizer
- paired-only vs weak-paired translation

Kill the idea if:

- generic time-series FMs match performance under identical splits
- Brain-OF matches translation and adaptation performance under the same leakage-proof benchmark
- gains vanish on held-out-site/dataset
- model mainly predicts subject/site identity
- translation works only on tiny paired datasets and fails weak-pair validation
- uncertainty is uncalibrated under missing modalities

## 10. Compute and Data Plan

Assume hundreds of H100s, but treat compute as the least interesting advantage.

Phase 1:

- Build data lake indexes for OpenNeuro, DANDI, NeuralBench, HCP, NSD/Algonauts, Allen/IBL, and controlled-access manifests for UKB/ABCD/ADNI.
- Standardize metadata before tensors.
- Every example gets provenance: dataset, site, subject, session, device/scanner, preprocessing, task, stimulus, split hash.
- Define v1 benchmark tasks before training: stimulus+past fMRI -> future fMRI, EEG/MEG -> shared state, fMRI -> EEG/MEG spectral proxy when paired, anatomy/fMRI -> subject latent, missing-modality reconstruction, few-shot adaptation.

Phase 2:

- Train modality specialists and baselines.
- Reproduce or wrap TRIBE v2, BrainVista, Brain-OF, BrainOmni, Brain Harmony where licenses and code allow; otherwise reproduce architecture-class baselines.
- Train tokenizers.
- Train shared translator at 300M-1B params.
- Scale only after leakage-proof debug suite passes.

Phase 3:

- Scale multi-timescale translator to 1B-5B params.
- Add weak-paired stimulus alignment and subject adaptation.
- Run full ablations and uncertainty calibration.

## 11. 90-Day Plan

Days 1-15:

- Freeze problem statement: leakage-proof Neural Translation benchmark and subject-adaptive translator.
- Build literature matrix and code/license registry.
- Implement dataset manifests and leakage-proof split engine.
- Reproduce/wrap TRIBE v2, BrainVista, Brain-OF, BrainOmni, Brain Harmony, NeuralBench, and NDT smoke baselines where licenses allow.

Days 16-30:

- Build universal event schema and token provenance.
- Implement fMRI, EEG/MEG, and spikes adapters.
- Create synthetic and tiny real-data smoke tasks.
- Establish held-out-subject, held-out-site, held-out-dataset split reports.
- Implement v1 task definitions: future fMRI, EEG/MEG shared-state, fMRI-to-spectral proxy, anatomy/fMRI subject state, missing modality, few-shot adaptation.

Days 31-50:

- Train modality-specific tokenizers.
- Implement Neural State Space Translator v0.
- Run forecasting and missing-modality tasks on small suites.
- Compare against Transformer, Mamba, TRIBE/BrainVista-style, Brain-OF-style, BrainOmni-style, EEGPT/LaBraM-style, LFADS/NDT-style baselines.

Days 51-70:

- Add few-shot subject adapters.
- Add weak-pair cross-modal alignment.
- Add TRIBE v2 and BrainVista-style stimulus-to-fMRI baselines and use them as constraints.
- Add Brain-OF as the primary multimodal boss fight.
- Run ablations at mid-scale.

Days 71-90:

- Full benchmark run on selected datasets.
- Write first paper draft.
- Package repo with configs, data manifests, model cards, and leakage reports.
- Decide go/no-go using kill criteria.

## 12. Risks and Weak Claims

Weak claim: "We built a brain foundation model."

Better claim: "We built a leakage-proof cross-modal neural translation benchmark and a model that improves missing-modality and future-state prediction under held-out-subject/site/dataset splits."

Weak claim: "We are the first multimodal neural foundation model."

Better claim: "We evaluate against the crowded multimodal field, especially TRIBE v2, BrainVista, Brain-OF, BrainOmni, and Brain Harmony, and win only on stricter translation/adaptation tasks."

Weak claim: "We invented universal neural tokenization."

Better claim: "We add provenance-aware, uncertainty-aware tokenization for translation benchmarks, building beyond BrainOmni/Brain-OF-style tokenizers."

Weak claim: "Digital twin of a human brain."

Better claim: "Research-grade neural-state simulator for stimulus-conditioned and modality-conditioned brain response hypotheses."

Weak claim: "Clinical diagnosis."

Better claim: "Secondary transfer to clinical labels, no treatment claims."

High-risk area:

- cross-modal translation without same-subject paired data.

Mitigation:

- use paired datasets only for validation,
- weak-paired stimulus alignment for training,
- teacher-student distillation,
- uncertainty calibration,
- report failures plainly.

## 13. Best Paper Title

Best:

**Neural Translation Under Leakage-Proof Splits**

Sharper:

**NeuroTwin: A Subject-Adaptive Neural Translator for Missing-Modality and Future-State Brain Prediction**

Avoid:

**A Digital Twin of the Human Brain**

Also avoid:

**The First Multimodal Brain Foundation Model**

**A Universal Neural Tokenizer**

That will invite overclaim scrutiny and deservedly so.

## 14. Abstract Draft

Brain foundation models have recently advanced fMRI, EEG/MEG, and intracortical modeling, with strong systems now covering stimulus-to-fMRI prediction and fMRI/EEG/MEG pretraining. Yet current progress remains difficult to compare because models are often evaluated on modality-specific tasks with inconsistent leakage controls, limited missing-modality tests, and weak few-shot subject adaptation protocols. We introduce NeuroTwin, a leakage-proof neural translation benchmark and subject-adaptive state-space architecture for shared latent brain-state modeling across heterogeneous recordings. NeuroTwin combines provenance-aware neural tokenization with a multi-timescale backbone that fuses slow spatial signals from fMRI/anatomy with fast temporal signals from EEG, MEG, and spiking activity where available. The model is trained with future-state prediction, missing-modality reconstruction, stimulus-conditioned response prediction, and subject-adaptation objectives. We evaluate under subject-held-out, site-held-out, dataset-held-out, modality-held-out, and time-held-out splits against TRIBE v2/BrainVista-style stimulus-to-fMRI models, Brain-OF-style multimodal foundation models, BrainOmni-style EEG/MEG tokenizers, and modality-specialist baselines. NeuroTwin's central claim is not being first; it is making cross-modal neural translation measurable, hard to cheat, and useful for research-grade in-silico experiments.

## 15. GitHub Repo Structure

```text
neurotwin/
  pyproject.toml
  README.md
  docs/
    research/
      neurotwin-technical-report.md
    licenses/
      upstream-registry.md
    methods/
      leakage-proof-evaluation.md
  configs/
    data/
    model/
    train/
    eval/
  src/neurotwin/
    data/
      schemas.py
      split_manifest.py
      leakage.py
      adapters/
        openneuro.py
        dandi.py
        neuralbench.py
        hcp.py
        nsd_algonauts.py
        allen_ibl.py
    tokenizers/
      fmri.py
      eeg_meg.py
      spikes.py
      anatomy.py
      stimulus.py
      provenance.py
    models/
      neural_state_space_translator.py
      neural_token_transformer.py
      causal_adapter.py
      baselines/
    training/
      objectives.py
      subject_adaptation.py
      losses.py
    eval/
      metrics.py
      benchmark_suites.py
      reports.py
    cli.py
  tests/
    test_schemas.py
    test_splits_no_leakage.py
    test_synthetic_adapters.py
    test_models_shapes.py
    test_metrics.py
```

Implementation plan changes:

1. Add TRIBE v2 and BrainVista as stimulus-to-fMRI baselines and design constraints.
2. Rename core architecture from generic "NeuroTwin model" to `NeuralStateSpaceTranslator`.
3. Treat Brain-OF as the primary multimodal boss fight.
4. Treat BrainOmni and Brain Harmony as tokenizer/representation competitors.
5. Promote leakage-proof `translation` benchmark suite before model scale and before clinical tasks.
6. Define v1 tasks explicitly: future fMRI, EEG/MEG shared-state mapping, fMRI-to-EEG/MEG spectral proxy where paired data exists, anatomy/fMRI subject latent, missing-modality reconstruction, few-shot subject adaptation.
7. Promote provenance/uncertainty tokenization as a component, not a "first universal tokenizer" claim.
8. Add `subject_adaptation_5_10_20_min` as a required benchmark.
9. Add `weak_pairing` and `paired_validation` data protocols.
10. Keep clinical labels secondary.

## 16. Reading List

Primary constraints:

- TRIBE v2: https://arxiv.org/abs/2605.04326
- TRIBE v2 Meta page: https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/
- TRIBE v2 model card: https://huggingface.co/facebook/tribev2
- TRIBE / Algonauts 2025 repo: https://github.com/facebookresearch/algonauts-2025
- BrainVista: https://arxiv.org/abs/2602.04512
- Multimodal Seq2Seq Transformer for naturalistic stimuli: https://arxiv.org/abs/2507.18104
- Brain-OF: https://arxiv.org/abs/2602.23410
- BrainOmni: https://arxiv.org/abs/2505.18185
- Brain Harmony / BrainHarmonix: https://arxiv.org/abs/2509.24693

fMRI/neuroimaging:

- BrainLM: https://openreview.net/forum?id=RwI7ZEfR27
- Brain-JEPA: https://papers.nips.cc/paper_files/paper/2024/file/9c3828adf1500f5de3c56f6550dfe43c-Paper-Conference.pdf
- NeuroSTORM: https://arxiv.org/abs/2506.11167
- Brain-DiT: https://arxiv.org/abs/2604.12683
- BrainGFM: https://arxiv.org/abs/2506.02044
- BrainSymphony: https://arxiv.org/abs/2506.18314

EEG/MEG:

- LaBraM: https://arxiv.org/abs/2405.18765
- NeuroLM: https://arxiv.org/abs/2409.00101
- EEGPT: https://github.com/BINE022/EEGPT
- NeuroAtlas: https://arxiv.org/abs/2605.14698
- NeuralBench: https://github.com/facebookresearch/neuroai/blob/main/neuralbench-repo/README.md

Spikes/population:

- NDT3: https://openreview.net/forum?id=ONOe6cAE9I
- NDT3 repo: https://github.com/joel99/ndt3
- Neuroformer: https://arxiv.org/abs/2311.00136
- Neuroformer repo: https://github.com/a-antoniades/Neuroformer
- CEBRA: https://www.nature.com/articles/s41586-023-06031-6
- LFADS: https://pmc.ncbi.nlm.nih.gov/articles/PMC6380887/
- OmniMouse: https://arxiv.org/abs/2604.18827

Digital twin/mechanistic:

- The Virtual Brain: https://www.thevirtualbrain.org/
- Virtual Epileptic Patient: https://www.medrxiv.org/content/10.1101/2022.01.19.22269404v1
- VEP cohort: https://pmc.ncbi.nlm.nih.gov/articles/PMC12043236/
- EU Neurotwin reporting: https://cordis.europa.eu/project/id/101017716/reporting
- EBRAINS: https://ebrains.eu/
- Digital Twin Brain review/platform: https://arxiv.org/abs/2308.01941 and https://arxiv.org/abs/2308.01241

Datasets/infrastructure:

- OpenNeuro: https://openneuro.org/
- BIDS: https://bids.neuroimaging.io/
- DANDI docs: https://docs.dandiarchive.org/introduction/
- DANDI GitHub scale snapshot: https://github.com/dandi
- HCP Young Adult: https://www.humanconnectome.org/study/hcp-young-adult/data-release/1200-subjects-data-release
- UK Biobank imaging: https://www.ukbiobank.ac.uk/enable-your-research/about-our-data/imaging-data
- ABCD: https://abcdstudy.org/about/
- NSD: https://naturalscenesdataset.org/
- Allen Brain Observatory: https://observatory.brain-map.org/visualcoding/
- IBL Brain-Wide Map: https://www.internationalbrainlab.com/brainwide-map
