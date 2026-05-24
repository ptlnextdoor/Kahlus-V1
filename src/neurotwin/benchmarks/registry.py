from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompetitorSpec:
    competitor_id: str
    display_name: str
    family: str
    role: str
    upstream_url: str
    license_status: str
    notes: str


def competitor_registry() -> tuple[CompetitorSpec, ...]:
    return (
        CompetitorSpec(
            competitor_id="tribe_v2",
            display_name="TRIBE v2",
            family="stimulus-to-fMRI",
            role="Primary baseline for video/audio/language -> fMRI prediction and in-silico neuroscience claims.",
            upstream_url="https://arxiv.org/abs/2605.04326",
            license_status="paper-first; verify code/weights before reuse",
            notes="Makes stimulus-to-brain prediction a crowded baseline, not NeuroTwin novelty.",
        ),
        CompetitorSpec(
            competitor_id="brainvista",
            display_name="BrainVista",
            family="naturalistic fMRI next-token prediction",
            role="Primary baseline for multimodal stimulus-conditioned fMRI rollout.",
            upstream_url="https://arxiv.org/abs/2602.04512",
            license_status="paper-first; verify code/weights before reuse",
            notes="Crowds future fMRI from stimulus and long-horizon rollout claims.",
        ),
        CompetitorSpec(
            competitor_id="brain_of",
            display_name="Brain-OF",
            family="fMRI/EEG/MEG foundation model",
            role="Boss-fight baseline for generic multimodal neural foundation modeling.",
            upstream_url="https://arxiv.org/abs/2602.23410",
            license_status="paper-first; verify code/weights before reuse",
            notes="Overlaps with universal tokenizer/translator; NeuroTwin must be stricter on translation benchmarks and held-out splits.",
        ),
        CompetitorSpec(
            competitor_id="brainomni",
            display_name="BrainOmni",
            family="EEG/MEG foundation model",
            role="Baseline for electrophysiology tokenization and sensor-layout generalization.",
            upstream_url="https://arxiv.org/abs/2505.18185",
            license_status="paper-first; verify code/weights before reuse",
            notes="Covers unified EEG/MEG tokenizer and unseen-device generalization.",
        ),
        CompetitorSpec(
            competitor_id="brain_harmony",
            display_name="Brain Harmony",
            family="structural MRI + fMRI",
            role="Baseline for compact anatomy/function brain tokens.",
            upstream_url="https://arxiv.org/abs/2509.24693",
            license_status="paper-first; verify code/weights before reuse",
            notes="Crowds structure + function tokenization.",
        ),
        CompetitorSpec(
            competitor_id="transformer",
            display_name="Transformer",
            family="general sequence model",
            role="Strong shared-data baseline.",
            upstream_url="https://arxiv.org/abs/1706.03762",
            license_status="implementation local/permissive",
            notes="Must use the exact same splits and inputs as NeuroTwin.",
        ),
        CompetitorSpec(
            competitor_id="mamba_ssm",
            display_name="Mamba/SSM",
            family="selective state-space model",
            role="Strong long-sequence dynamics baseline.",
            upstream_url="https://github.com/state-spaces/mamba",
            license_status="permissive upstream; pin commit before vendoring",
            notes="Baseline and possible NeuroTwin core, but not by itself a novelty claim.",
        ),
        CompetitorSpec(
            competitor_id="modality_specialist",
            display_name="Modality-specialist baselines",
            family="specialist models",
            role="NDT3/Neuroformer for spikes, Braindecode/EEGNet-style EEG, fMRI-specific baselines.",
            upstream_url="https://github.com/joel99/ndt3",
            license_status="mixed; quarantine restricted adapters",
            notes="Prevents weak generalist comparisons.",
        ),
    )
