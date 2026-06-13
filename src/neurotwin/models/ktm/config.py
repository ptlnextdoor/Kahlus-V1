"""Configuration for the Kahlus v3 KTM (Kahlus Transition Model) scaffold.

PROPOSED / SYNTHETIC ONLY. Minimal numpy scaffolding to exercise the Transition Gym objects
(history encoder, memory, response profile C_K, perturbation operators T_a, commutators
[Ta,Tb], expert readout, uncertainty). Not a built model, not a scientific claim.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KTMConfig:
    seed: int = 0
    history_len: int = 6
    eeg_channels: int = 5
    n_perturbations: int = 4  # K
    horizon: int = 5  # H
    embed_dim: int = 16
    memory_dim: int = 12
    n_experts: int = 4
    memory_rho: float = 0.8
    uncertainty_floor: float = 1e-3

    def validate(self) -> "KTMConfig":
        positives = {
            "history_len": self.history_len,
            "eeg_channels": self.eeg_channels,
            "n_perturbations": self.n_perturbations,
            "horizon": self.horizon,
            "embed_dim": self.embed_dim,
            "memory_dim": self.memory_dim,
            "n_experts": self.n_experts,
        }
        for name, value in positives.items():
            if int(value) < 1:
                raise ValueError(f"KTMConfig.{name} must be >= 1, got {value}")
        if not 0.0 <= self.memory_rho < 1.0:
            raise ValueError("KTMConfig.memory_rho must be in [0, 1)")
        if self.uncertainty_floor <= 0.0:
            raise ValueError("KTMConfig.uncertainty_floor must be positive")
        return self
