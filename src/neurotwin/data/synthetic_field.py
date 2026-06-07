from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SyntheticLatentFieldSample:
    latent_field: np.ndarray
    fmri: np.ndarray
    eeg: np.ndarray
    stimulus: np.ndarray
    metadata: dict[str, Any]


def generate_synthetic_latent_field(
    *,
    seed: int = 0,
    n_samples: int = 32,
    time_steps: int = 8,
    n_nodes: int = 6,
    latent_dim: int = 5,
    eeg_channels: int = 4,
    stimulus_dim: int = 3,
    noise_scale: float = 0.03,
) -> SyntheticLatentFieldSample:
    """Generate a synthetic field with fMRI-like and EEG-like observations."""

    if min(n_samples, time_steps, n_nodes, latent_dim, eeg_channels, stimulus_dim) < 1:
        raise ValueError("all synthetic field dimensions must be positive")
    rng = np.random.default_rng(seed)
    stimulus = rng.normal(size=(n_samples, time_steps, stimulus_dim)).astype(np.float32)
    node_bias = rng.normal(scale=0.2, size=(n_nodes, latent_dim)).astype(np.float32)
    dynamics = rng.normal(scale=0.12, size=(latent_dim, latent_dim)).astype(np.float32)
    stimulus_drive = rng.normal(scale=0.18, size=(stimulus_dim, latent_dim)).astype(np.float32)
    coupling_left = rng.normal(scale=0.12, size=(n_nodes, 2)).astype(np.float32)
    coupling = coupling_left @ coupling_left.T
    coupling = coupling / max(float(np.max(np.abs(coupling))), 1.0)
    field: np.ndarray = np.zeros((n_samples, time_steps, n_nodes, latent_dim), dtype=np.float32)
    field[:, 0] = rng.normal(scale=0.2, size=(n_samples, n_nodes, latent_dim)).astype(np.float32) + node_bias
    for step in range(1, time_steps):
        previous = field[:, step - 1]
        recurrent = np.tanh(previous @ dynamics)
        graph_message = np.einsum("ij,bjd->bid", coupling, previous) / float(n_nodes)
        drive = stimulus[:, step] @ stimulus_drive
        field[:, step] = 0.74 * previous + 0.18 * recurrent + 0.08 * graph_message + drive[:, None, :] + node_bias
    field += rng.normal(scale=noise_scale, size=field.shape).astype(np.float32)
    fmri_readout = rng.normal(scale=0.35, size=(latent_dim,)).astype(np.float32)
    raw_fmri = np.einsum("btnd,d->btn", field, fmri_readout)
    lagged = np.concatenate([raw_fmri[:, :1], raw_fmri[:, :-1]], axis=1)
    fmri = (0.65 * raw_fmri + 0.35 * lagged + rng.normal(scale=noise_scale, size=raw_fmri.shape)).astype(np.float32)
    eeg_projection = rng.normal(scale=0.25, size=(n_nodes * latent_dim, eeg_channels)).astype(np.float32)
    eeg_flat = field.reshape(n_samples, time_steps, n_nodes * latent_dim)
    eeg = np.tanh(eeg_flat @ eeg_projection)
    eeg += rng.normal(scale=noise_scale, size=eeg.shape).astype(np.float32)
    metadata = {
        "seed": int(seed),
        "claim_status": "synthetic_plumbing_only",
        "field_shape": list(field.shape),
        "fmri_shape": list(fmri.shape),
        "eeg_shape": list(eeg.shape),
        "stimulus_shape": list(stimulus.shape),
    }
    return SyntheticLatentFieldSample(
        latent_field=field.astype(np.float32),
        fmri=fmri.astype(np.float32),
        eeg=eeg.astype(np.float32),
        stimulus=stimulus.astype(np.float32),
        metadata=metadata,
    )
