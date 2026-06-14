"""Kahlus v3 KTM (Kahlus Transition Model) scaffold (PROPOSED / SYNTHETIC ONLY).

Minimal numpy scaffolding for the Transition Gym: history encoder, memory, finite response
profile C_K, latent perturbation operators T_a, their commutators [Ta,Tb], a mixture-of-experts
readout, and a per-perturbation uncertainty head. Not a built model, not a scientific claim.
"""

from __future__ import annotations

from neurotwin.models.ktm.config import KTMConfig
from neurotwin.models.ktm.ktm import KTM
from neurotwin.models.ktm.torch_ktm import TorchKTM, TorchKTMConfig

__all__ = ["KTMConfig", "KTM", "TorchKTM", "TorchKTMConfig"]
