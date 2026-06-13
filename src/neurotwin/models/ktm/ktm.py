"""KTM (Kahlus Transition Model) orchestrator scaffold.

Wires the history encoder, memory, response profile C_K, latent perturbation operators T_a,
their commutators [Ta,Tb], an optional mixture-of-experts readout, and a per-perturbation
uncertainty head. Pure numpy and deterministic for a given seed.

PROPOSED / SYNTHETIC ONLY. No training, no scientific claim.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from neurotwin.numerics import ignore_spurious_matmul_warnings
from neurotwin.models.ktm.config import KTMConfig
from neurotwin.models.ktm.history_encoder import HistoryEncoder
from neurotwin.models.ktm.lie_generators import commutator_matrix
from neurotwin.models.ktm.memory import Memory
from neurotwin.models.ktm.neuro_experts import NeuroExperts
from neurotwin.models.ktm.perturbation_operators import PerturbationOperators
from neurotwin.models.ktm.response_profile import ResponseProfileHead
from neurotwin.models.ktm.uncertainty import UncertaintyHead


class KTM:
    def __init__(self, config: KTMConfig) -> None:
        self.config = config.validate()
        rng = np.random.default_rng(self.config.seed)
        cfg = self.config
        self.encoder = HistoryEncoder(rng, cfg.history_len, cfg.eeg_channels, cfg.embed_dim)
        self.memory = Memory(rng, cfg.embed_dim, cfg.memory_dim, cfg.memory_rho)
        self.operators = PerturbationOperators(rng, cfg.memory_dim, cfg.n_perturbations)
        self.response_head = ResponseProfileHead(
            rng, cfg.memory_dim, cfg.n_perturbations, cfg.horizon, cfg.eeg_channels
        )
        self.experts = NeuroExperts(rng, cfg.memory_dim, cfg.n_experts, cfg.horizon, cfg.eeg_channels)
        self.uncertainty = UncertaintyHead(rng, cfg.memory_dim, cfg.n_perturbations, cfg.uncertainty_floor)

    def encode_memory(self, history: np.ndarray) -> np.ndarray:
        with ignore_spurious_matmul_warnings():
            return self.memory.project(self.encoder.encode(history))

    def predict_response_profile(self, history: np.ndarray) -> np.ndarray:
        """C_K(h_t): ``(batch, K, horizon, eeg_channels)``."""

        with ignore_spurious_matmul_warnings():
            return self.response_head.predict(self.encode_memory(history))

    def predict_future(self, history: np.ndarray, perturbation_index: int) -> np.ndarray:
        """pθ(τ | h_t, a_k): ``(batch, horizon, eeg_channels)``."""

        profile = self.predict_response_profile(history)
        return profile[:, int(perturbation_index)]

    def predict_aggregate(self, history: np.ndarray) -> np.ndarray:
        """Mixture-of-experts aggregate forecast: ``(batch, horizon, eeg_channels)``."""

        with ignore_spurious_matmul_warnings():
            return self.experts.predict(self.encode_memory(history))

    def predict_uncertainty(self, history: np.ndarray) -> np.ndarray:
        """Per-perturbation positive variances: ``(batch, K)``."""

        with ignore_spurious_matmul_warnings():
            return self.uncertainty.predict(self.encode_memory(history))

    def commutators(self) -> np.ndarray:
        """Pairwise commutator-norm matrix of the latent operators (``(K, K)``)."""

        return commutator_matrix(self.operators.operators())

    def metadata(self) -> dict[str, Any]:
        comm = self.commutators()
        return {
            "branch": "v3",
            "claim_status": "synthetic_scaffold_only",
            "seed": int(self.config.seed),
            "n_perturbations": int(self.config.n_perturbations),
            "horizon": int(self.config.horizon),
            "max_commutator_norm": float(np.max(comm)),
            "non_commutative": bool(float(np.max(comm)) > 1e-8),
        }
