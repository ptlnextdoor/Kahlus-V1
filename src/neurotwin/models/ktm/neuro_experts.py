"""Mixture-of-neuro-experts readout for the KTM scaffold.

A softmax gate over the memory state mixes ``n_experts`` linear trajectory experts into a
single aggregated forecast pθ(τ | h_t). This is a small, optional readout used as an
alternative/aggregate to the per-perturbation response profile.
"""

from __future__ import annotations

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class NeuroExperts:
    def __init__(
        self,
        rng: np.random.Generator,
        memory_dim: int,
        n_experts: int,
        horizon: int,
        eeg_channels: int,
    ) -> None:
        self.memory_dim = int(memory_dim)
        self.n_experts = int(n_experts)
        self.horizon = int(horizon)
        self.eeg_channels = int(eeg_channels)
        out_dim = self.horizon * self.eeg_channels
        self.gate = rng.normal(scale=1.0 / np.sqrt(self.memory_dim), size=(self.memory_dim, self.n_experts))
        self.experts = rng.normal(scale=1.0 / np.sqrt(self.memory_dim), size=(self.n_experts, self.memory_dim, out_dim))

    def gate_weights(self, memory: np.ndarray) -> np.ndarray:
        return _softmax(np.asarray(memory, dtype=np.float64) @ self.gate)

    def predict(self, memory: np.ndarray) -> np.ndarray:
        """``memory`` is ``(batch, memory_dim)`` -> ``(batch, horizon, eeg_channels)``."""

        memory = np.asarray(memory, dtype=np.float64)
        weights = self.gate_weights(memory)  # (batch, n_experts)
        expert_out = np.einsum("bm,emo->beo", memory, self.experts)  # (batch, n_experts, out_dim)
        mixed = np.einsum("be,beo->bo", weights, expert_out)
        return mixed.reshape(memory.shape[0], self.horizon, self.eeg_channels)
