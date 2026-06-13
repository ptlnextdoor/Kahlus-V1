"""Dual-field compiler: builds seeded parameters and rolls out the synthetic system.

PROPOSED / SYNTHETIC ONLY. Produces deterministic fast/slow field trajectories and their
EEG-like / BOLD-like observations for local falsification and baseline comparison. No
scientific claim is implied by running it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from neurotwin.numerics import ignore_spurious_matmul_warnings
from neurotwin.models.dual_field.config import DualFieldConfig
from neurotwin.models.dual_field.coupling import hrf_lag_weights, low_rank_coupling
from neurotwin.models.dual_field.fast_field import FastFieldParams, fast_field_step
from neurotwin.models.dual_field.observation_heads import ObservationHeads
from neurotwin.models.dual_field.slow_field import SlowFieldParams, slow_field_step
from neurotwin.models.dual_field.stability import assert_finite, is_stable, spectral_radius


@dataclass(frozen=True)
class DualFieldRollout:
    config: DualFieldConfig
    neural: np.ndarray  # (n_samples, time_steps, neural_dim)
    hemo: np.ndarray  # (n_samples, time_steps, hemo_dim)
    eeg: np.ndarray  # (n_samples, time_steps, eeg_channels)
    bold: np.ndarray  # (n_samples, time_steps, bold_channels)
    stimulus: np.ndarray  # (n_samples, time_steps, stimulus_dim)
    metadata: dict[str, Any]


class DualFieldCompiler:
    """Builds the dual-field parameters from a config and simulates trajectories."""

    def __init__(self, config: DualFieldConfig) -> None:
        self.config = config.validate()
        rng = np.random.default_rng(self.config.seed)
        cfg = self.config

        decay = rng.uniform(cfg.decay_low, cfg.decay_high, size=cfg.neural_dim)
        coupling = low_rank_coupling(rng, cfg.neural_dim, cfg.coupling_rank, cfg.coupling_radius)
        stimulus_in = rng.normal(scale=0.5, size=(cfg.stimulus_dim, cfg.neural_dim))
        residual_w = rng.normal(scale=0.3, size=(cfg.neural_dim, cfg.neural_dim))
        self.fast_params = FastFieldParams(
            decay=decay,
            coupling=coupling,
            stimulus_in=stimulus_in,
            residual_w=residual_w,
            residual_scale=cfg.residual_scale,
            dt=cfg.dt,
            state_clip=cfg.state_clip,
        )
        self.slow_params = SlowFieldParams(
            rho=cfg.rho,
            readout=rng.normal(scale=0.5, size=(cfg.neural_dim, cfg.hemo_dim)),
            lag_weights=hrf_lag_weights(cfg.hemo_lag),
            state_clip=cfg.state_clip,
        )
        self.heads = ObservationHeads(
            eeg_w=rng.normal(scale=0.5, size=(cfg.neural_dim, cfg.eeg_channels)),
            bold_w=rng.normal(scale=0.5, size=(cfg.hemo_dim, cfg.bold_channels)),
        )
        self._rng = rng

    def rollout(self) -> DualFieldRollout:
        cfg = self.config
        rng = self._rng
        b, t, dn, dh = cfg.n_samples, cfg.time_steps, cfg.neural_dim, cfg.hemo_dim
        window = cfg.hemo_lag + 1

        stimulus = (cfg.stimulus_scale * rng.normal(size=(b, t, cfg.stimulus_dim))).astype(np.float64)
        eeg_noise = cfg.noise_scale * rng.normal(size=(b, t, cfg.eeg_channels))
        bold_noise = cfg.noise_scale * rng.normal(size=(b, t, cfg.bold_channels))
        hemo_proc_noise = cfg.noise_scale * rng.normal(size=(b, t, dh))

        neural = np.zeros((b, t, dn), dtype=np.float64)
        hemo = np.zeros((b, t, dh), dtype=np.float64)
        eeg = np.zeros((b, t, cfg.eeg_channels), dtype=np.float64)
        bold = np.zeros((b, t, cfg.bold_channels), dtype=np.float64)

        n_state = rng.normal(scale=0.2, size=(b, dn))
        h_state = np.zeros((b, dh), dtype=np.float64)
        history: list[np.ndarray] = []

        with ignore_spurious_matmul_warnings():
            for step in range(t):
                neural[:, step] = n_state
                hemo[:, step] = h_state
                eeg[:, step] = self.heads.observe_eeg(n_state) + eeg_noise[:, step]
                bold[:, step] = self.heads.observe_bold(h_state) + bold_noise[:, step]

                history.append(n_state)
                if step < t - 1:
                    recent = history[-window:]
                    if len(recent) < window:
                        recent = [recent[0]] * (window - len(recent)) + recent
                    neural_window = np.stack(recent, axis=1)  # (b, window, dn)
                    h_state = slow_field_step(h_state, neural_window, hemo_proc_noise[:, step], self.slow_params)
                    n_state = fast_field_step(n_state, stimulus[:, step], self.fast_params)

        for name, array in (("neural", neural), ("hemo", hemo), ("eeg", eeg), ("bold", bold)):
            assert_finite(array, name)

        radius = spectral_radius(self.fast_params.coupling)
        metadata = {
            "seed": int(cfg.seed),
            "claim_status": "synthetic_scaffold_only",
            "branch": "v2",
            "neural_shape": list(neural.shape),
            "hemo_shape": list(hemo.shape),
            "eeg_shape": list(eeg.shape),
            "bold_shape": list(bold.shape),
            "stimulus_shape": list(stimulus.shape),
            "fast_coupling_spectral_radius": radius,
            "fast_coupling_stable": bool(is_stable(self.fast_params.coupling)),
        }
        return DualFieldRollout(
            config=cfg,
            neural=neural.astype(np.float32),
            hemo=hemo.astype(np.float32),
            eeg=eeg.astype(np.float32),
            bold=bold.astype(np.float32),
            stimulus=stimulus.astype(np.float32),
            metadata=metadata,
        )


def simulate_dual_field(config: DualFieldConfig | None = None) -> DualFieldRollout:
    """Convenience: build a compiler and return a single rollout."""

    return DualFieldCompiler(config or DualFieldConfig()).rollout()
