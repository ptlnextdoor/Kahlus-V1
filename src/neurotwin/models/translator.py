from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NeuralStateSpaceTranslatorConfig:
    """Configuration shell for the future NeuroTwin model implementation."""

    latent_dim: int = 512
    ssm_layers: int = 12
    adapter_rank: int = 16
    modalities: tuple[str, ...] = ("fmri", "eeg", "meg", "spikes", "behavior", "stimulus", "anatomy")
    objectives: tuple[str, ...] = (
        "missing_modality_reconstruction",
        "future_state_forecasting",
        "few_shot_subject_adaptation",
    )

    def describe(self) -> str:
        modalities = ", ".join(self.modalities)
        objectives = ", ".join(self.objectives)
        return (
            f"NeuralStateSpaceTranslator(latent_dim={self.latent_dim}, "
            f"ssm_layers={self.ssm_layers}, adapter_rank={self.adapter_rank}, "
            f"modalities=[{modalities}], objectives=[{objectives}])"
        )
