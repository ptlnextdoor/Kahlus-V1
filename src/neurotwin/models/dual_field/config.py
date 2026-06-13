"""Configuration for the Kahlus v2 synthetic dual-field scaffold.

PROPOSED / SYNTHETIC ONLY. This is a local falsification scaffold, not a built model and
not a scientific claim. See ``docs/roadmap/kahlus_implementation_status.md``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DualFieldConfig:
    """Shapes and dynamics constants for the dual-field synthetic system.

    The system couples a *fast* neural field ``N`` (EEG-like readout) and a *slow*
    hemodynamic/measurement field ``H`` (BOLD/fNIRS-like readout):

        N_{t+1} = N_t + dt[-Λ N_t + K σ(N_t) + B U_t + Rθ(N_t, U_t)]
        H_{t+1} = ρ H_t + (1-ρ) gθ(N_{t-L:t}) + noise
        Y_EEG   = O_EEG(N_t)
        Y_BOLD  = O_BOLD(H_t)
    """

    seed: int = 0
    n_samples: int = 16
    time_steps: int = 24
    neural_dim: int = 8
    hemo_dim: int = 6
    eeg_channels: int = 5
    bold_channels: int = 4
    stimulus_dim: int = 3
    dt: float = 0.1
    decay_low: float = 0.6
    decay_high: float = 1.4
    coupling_rank: int = 2
    coupling_radius: float = 0.9
    residual_scale: float = 0.05
    stimulus_scale: float = 0.3
    rho: float = 0.9
    hemo_lag: int = 4
    state_clip: float = 50.0
    noise_scale: float = 0.02

    def validate(self) -> "DualFieldConfig":
        positives = {
            "n_samples": self.n_samples,
            "time_steps": self.time_steps,
            "neural_dim": self.neural_dim,
            "hemo_dim": self.hemo_dim,
            "eeg_channels": self.eeg_channels,
            "bold_channels": self.bold_channels,
            "stimulus_dim": self.stimulus_dim,
            "coupling_rank": self.coupling_rank,
            "hemo_lag": self.hemo_lag,
        }
        for name, value in positives.items():
            if int(value) < 1:
                raise ValueError(f"DualFieldConfig.{name} must be >= 1, got {value}")
        if self.dt <= 0.0:
            raise ValueError("DualFieldConfig.dt must be positive")
        if not 0.0 <= self.rho < 1.0:
            raise ValueError("DualFieldConfig.rho must be in [0, 1)")
        if self.coupling_radius <= 0.0:
            raise ValueError("DualFieldConfig.coupling_radius must be positive")
        if self.state_clip <= 0.0:
            raise ValueError("DualFieldConfig.state_clip must be positive")
        if self.hemo_lag >= self.time_steps:
            raise ValueError("DualFieldConfig.hemo_lag must be smaller than time_steps")
        return self
