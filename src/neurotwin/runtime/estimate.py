from __future__ import annotations

from typing import Any


def estimate_config(config: dict[str, Any]) -> dict[str, int | float | str]:
    model = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    latent_dim = int(model.get("latent_dim", 128))
    n_layers = int(model.get("n_layers", 2))
    input_dim = int(model.get("input_dim", 16))
    output_dim = int(model.get("output_dim", input_dim))
    estimated_parameters = (input_dim * latent_dim) + (latent_dim * latent_dim * max(n_layers, 1) * 3) + (latent_dim * output_dim)
    activation_mb = float(config.get("batch_size", 8)) * float(config.get("window_size", 128)) * latent_dim * 4 / (1024 * 1024)
    return {
        "estimated_parameters": int(estimated_parameters),
        "estimated_activation_mb": round(activation_mb, 3),
        "precision": str(config.get("precision", "fp32")),
    }
