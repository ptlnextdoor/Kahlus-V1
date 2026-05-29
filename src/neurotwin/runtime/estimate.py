from __future__ import annotations

from typing import Any


def estimate_config(config: dict[str, Any]) -> dict[str, int | float | str]:
    model = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    training = config.get("training", {}) if isinstance(config.get("training", {}), dict) else {}
    latent_dim = int(model.get("latent_dim", 128))
    n_layers = int(model.get("n_layers", 2))
    input_dim = int(model.get("input_dim", 16))
    output_dim = int(model.get("output_dim", input_dim))
    modalities = model.get("modalities", ["generic"])
    if not isinstance(modalities, list) or not modalities:
        modalities = ["generic"]
    backbone = str(model.get("backbone", "ssm_fallback"))
    encoder = str(model.get("encoder", "auto"))
    batch_size = int(config.get("batch_size", 8))
    window_size = int(config.get("window_size", config.get("window_length", 128)))
    grad_accum = int(config.get("gradient_accumulation_steps", training.get("gradient_accumulation_steps", 1)))
    precision = str(config.get("precision", training.get("precision", "fp32"))).lower()
    gradient_checkpointing = bool(
        config.get("gradient_checkpointing", model.get("gradient_checkpointing", training.get("gradient_checkpointing", False)))
    )
    bytes_per_value = 2 if precision in {"bf16", "fp16", "float16", "bfloat16"} else 4
    encoder_params = len(modalities) * ((input_dim * latent_dim) + (latent_dim * latent_dim if encoder in {"auto", "temporal_conv", "conv"} else 0))
    if backbone == "transformer":
        backbone_params = n_layers * (12 * latent_dim * latent_dim)
    else:
        backbone_params = n_layers * (6 * latent_dim * latent_dim)
    head_params = len(modalities) * latent_dim * output_dim * 3
    adapter_params = int(model.get("subject_adapter_dim", 0)) * latent_dim * 2
    estimated_parameters = encoder_params + backbone_params + head_params + adapter_params
    activation_mb = batch_size * window_size * latent_dim * bytes_per_value * max(n_layers, 1) / (1024 * 1024)
    optimizer_mb = estimated_parameters * 8 / (1024 * 1024)
    checkpoint_mb = estimated_parameters * bytes_per_value / (1024 * 1024)
    return {
        "estimated_parameters": int(estimated_parameters),
        "estimated_activation_mb": round(activation_mb, 3),
        "estimated_optimizer_mb": round(optimizer_mb, 3),
        "estimated_checkpoint_mb": round(checkpoint_mb, 3),
        "effective_batch_size": batch_size * max(grad_accum, 1),
        "backbone": backbone,
        "encoder": encoder,
        "precision": precision,
        "gradient_accumulation_steps": max(grad_accum, 1),
        "gradient_checkpointing": str(gradient_checkpointing),
        "compile": str(bool(training.get("compile", False))),
    }
