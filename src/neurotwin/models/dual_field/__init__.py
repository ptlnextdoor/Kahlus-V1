"""Kahlus v2 synthetic dual-field scaffold (PROPOSED / SYNTHETIC ONLY).

A small, local, deterministic dual-field system: a fast neural field N (EEG-like readout)
coupled to a slow hemodynamic/measurement field H (BOLD/fNIRS-like readout). Used for shape
stability, finite-output, reproducibility, and baseline-comparison falsification — not a
built model and not a scientific claim.
"""

from __future__ import annotations

from neurotwin.models.dual_field.config import DualFieldConfig
from neurotwin.models.dual_field.dual_field_compiler import (
    DualFieldCompiler,
    DualFieldRollout,
    simulate_dual_field,
)

__all__ = [
    "DualFieldConfig",
    "DualFieldCompiler",
    "DualFieldRollout",
    "simulate_dual_field",
    "run_v2_benchmark",
    "write_v2_report",
]


def __getattr__(name):  # lazy to avoid importing torch-free benchmark deps at package load
    if name in {"run_v2_benchmark", "write_v2_report"}:
        from neurotwin.models.dual_field import benchmark

        return getattr(benchmark, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
