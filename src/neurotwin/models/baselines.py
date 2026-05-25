from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class BaselineSpec:
    model_id: str
    display_name: str
    required_inputs: tuple[str, ...]
    supported_tasks: tuple[str, ...]
    notes: str


def baseline_registry() -> tuple[BaselineSpec, ...]:
    return (
        BaselineSpec(
            model_id="transformer",
            display_name="Transformer",
            required_inputs=("neural_tokens", "modality_tokens", "time_tokens"),
            supported_tasks=("future_state", "missing_modality"),
            notes="Shared data and splits; no weaker toy Transformer comparison.",
        ),
        BaselineSpec(
            model_id="mamba_ssm",
            display_name="Mamba/SSM",
            required_inputs=("neural_tokens", "time_tokens"),
            supported_tasks=("future_state", "long_rollout"),
            notes="Pin upstream state-spaces/mamba before heavy training.",
        ),
        BaselineSpec(
            model_id="modality_specialist",
            display_name="Modality-specialist",
            required_inputs=("modality_native_signal",),
            supported_tasks=("specialist_decode", "specialist_forecast"),
            notes="NDT3/Neuroformer/Braindecode/fMRI-specific baselines by modality.",
        ),
    )


class NumpyRidgeBaseline:
    """Closed-form ridge baseline for sanity checks and tiny CPU benchmarks."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyRidgeBaseline":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("x and y must be 2D arrays")
        xtx = x.T @ x
        reg = self.alpha * np.eye(xtx.shape[0])
        self.coef_ = np.linalg.solve(xtx + reg, x.T @ y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("NumpyRidgeBaseline must be fit before predict")
        return np.asarray(x, dtype=float) @ self.coef_


class TorchMLPBaseline(nn.Module):
    """Per-timepoint MLP baseline for neural windows."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchTCNBaseline(nn.Module):
    """Small Conv1D/TCN-style baseline with stable [batch, time, feature] IO."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected [batch, time, features]")
        return self.net(x.transpose(1, 2)).transpose(1, 2)
