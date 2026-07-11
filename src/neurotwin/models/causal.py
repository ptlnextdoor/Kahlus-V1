from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Module):
    """One-dimensional convolution whose output at t only uses inputs <= t."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or dilation < 1:
            raise ValueError("kernel_size and dilation must be positive")
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("CausalConv1d expects [batch, channels, time]")
        return self.conv(F.pad(values, (self.left_padding, 0)))


class CausalHRFAdapter(nn.Module):
    """Causal learned temporal response with an explicit nonnegative delay."""

    def __init__(self, channels: int, *, delay_steps: int = 1, kernel_size: int = 3) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if delay_steps < 0:
            raise ValueError("delay_steps cannot be negative")
        self.delay_steps = int(delay_steps)
        self.response = CausalConv1d(channels, channels, kernel_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("CausalHRFAdapter expects [batch, channels, time]")
        delayed = values
        if self.delay_steps:
            delayed = F.pad(values, (self.delay_steps, 0))[..., : values.shape[-1]]
        return self.response(delayed)


class CausalTransformerEncoder(nn.Module):
    """Transformer encoder with sinusoidal positions and an enforced causal mask."""

    def __init__(
        self,
        latent_dim: int,
        *,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if latent_dim < 1 or n_heads < 1 or n_layers < 1:
            raise ValueError("latent_dim, n_heads, and n_layers must be positive")
        if latent_dim % n_heads != 0:
            raise ValueError("latent_dim must be divisible by n_heads")
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.num_layers = int(n_layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("CausalTransformerEncoder expects [batch, time, latent_dim]")
        positioned = values + sinusoidal_positions(
            values.shape[1],
            values.shape[2],
            device=values.device,
            dtype=values.dtype,
        ).unsqueeze(0)
        mask = torch.triu(
            torch.ones(values.shape[1], values.shape[1], dtype=torch.bool, device=values.device),
            diagonal=1,
        )
        return self.encoder(positioned, mask=mask)


def sinusoidal_positions(
    length: int,
    latent_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if length < 1 or latent_dim < 1:
        raise ValueError("length and latent_dim must be positive")
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    pair_count = (latent_dim + 1) // 2
    scale = torch.exp(
        torch.arange(pair_count, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / max(1, pair_count - 1))
    )
    angles = position * scale.unsqueeze(0)
    encoded = torch.zeros(length, latent_dim, device=device, dtype=torch.float32)
    encoded[:, 0::2] = torch.sin(angles[:, : encoded[:, 0::2].shape[1]])
    encoded[:, 1::2] = torch.cos(angles[:, : encoded[:, 1::2].shape[1]])
    return encoded.to(dtype=dtype)
