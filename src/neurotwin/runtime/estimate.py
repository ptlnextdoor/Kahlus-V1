from __future__ import annotations

from neurotwin.config_types import PreparedTrainingConfigInput, resolve_prepared_config


def estimate_config(config: PreparedTrainingConfigInput) -> dict[str, int | float | str]:
    resolved = resolve_prepared_config(config, require_manifests=False, window_length_default=128)
    model = resolved.model
    runtime = resolved.runtime
    batch_size = runtime.batch_size or 8
    grad_accum = runtime.gradient_accumulation_steps
    precision = runtime.precision
    bytes_per_value = 2 if precision in {"bf16", "fp16", "float16", "bfloat16"} else 4
    encoder_params = len(model.modalities) * (
        (model.input_dim * model.latent_dim)
        + (model.latent_dim * model.latent_dim if model.encoder in {"auto", "temporal_conv", "conv"} else 0)
    )
    if model.backbone == "transformer":
        backbone_params = model.n_layers * (12 * model.latent_dim * model.latent_dim)
    else:
        backbone_params = model.n_layers * (6 * model.latent_dim * model.latent_dim)
    head_params = len(model.modalities) * model.latent_dim * model.output_dim * 3
    adapter_params = model.subject_adapter_dim * model.latent_dim * 2
    pair_operator_params = 0
    normalized_type = model.type.strip().lower().replace("-", "_")
    if normalized_type in {"neurotwin_pair_operator", "neurotwinpairoperator", "pair_operator", "ntp_o"}:
        pair_operator_params = model.output_dim * model.pair_rank * 2 + model.latent_dim * model.latent_dim
    nfc_params = 0
    if normalized_type in {"neurotwin_nfc", "nfc", "neural_field_compiler", "neuralfieldcompiler", "field_compiler"}:
        nfc_params = model.output_dim * model.pair_rank * 2 + model.latent_dim * model.latent_dim * 3
    estimated_parameters = encoder_params + backbone_params + head_params + adapter_params
    estimated_parameters += pair_operator_params + nfc_params
    model_status = "experimental_architecture" if nfc_params else "local_baseline"
    activation_mb = batch_size * resolved.window_length * model.latent_dim * bytes_per_value * max(model.n_layers, 1) / (1024 * 1024)
    optimizer_mb = estimated_parameters * 8 / (1024 * 1024)
    checkpoint_mb = estimated_parameters * bytes_per_value / (1024 * 1024)
    return {
        "estimated_parameters": int(estimated_parameters),
        "estimated_activation_mb": round(activation_mb, 3),
        "estimated_optimizer_mb": round(optimizer_mb, 3),
        "estimated_checkpoint_mb": round(checkpoint_mb, 3),
        "effective_batch_size": batch_size * max(grad_accum, 1),
        "model_type": model.type,
        "model_status": model_status,
        "backbone": model.backbone,
        "encoder": model.encoder,
        "precision": precision,
        "gradient_accumulation_steps": max(grad_accum, 1),
        "gradient_checkpointing": str(model.gradient_checkpointing),
        "compile": str(runtime.compile),
        "use_pair_kernel": str(model.use_pair_kernel),
        "use_observation_operator": str(model.use_observation_operator),
        "use_uncertainty": str(model.use_uncertainty),
    }
