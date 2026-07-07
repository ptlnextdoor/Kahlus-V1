from __future__ import annotations

from neurotwin.config_types import PreparedTrainingConfigInput, resolve_prepared_config
from neurotwin.models.architecture_registry import architecture_status, estimate_architecture_extra_parameters, normalize_architecture_type


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
    pair_state_factor_values = 0
    model_type = normalize_architecture_type(model.type)
    pair_operator_model = model_type == "NeuroTwinPairOperator"
    if pair_operator_model:
        pair_state_factor_values = model.output_dim * model.pair_rank * 2
    estimated_parameters = encoder_params + backbone_params + head_params + adapter_params
    estimated_parameters += estimate_architecture_extra_parameters(model)
    model_status = architecture_status(model.type)
    activation_mb = batch_size * resolved.window_length * model.latent_dim * bytes_per_value * max(model.n_layers, 1) / (1024 * 1024)
    optimizer_mb = estimated_parameters * 8 / (1024 * 1024)
    checkpoint_mb = estimated_parameters * bytes_per_value / (1024 * 1024)
    pair_state_mb = pair_state_factor_values * bytes_per_value / (1024 * 1024)
    single_a100_mb = activation_mb + optimizer_mb + checkpoint_mb
    seven_a100_ddp_mb = activation_mb + optimizer_mb + checkpoint_mb
    return {
        "estimated_parameters": int(estimated_parameters),
        "estimated_activation_mb": round(activation_mb, 3),
        "estimated_optimizer_mb": round(optimizer_mb, 3),
        "estimated_checkpoint_mb": round(checkpoint_mb, 3),
        "estimated_pair_state_mb": round(pair_state_mb, 3),
        "estimated_1xa100_runtime_mb": round(single_a100_mb, 3),
        "estimated_7xa100_ddp_per_gpu_mb": round(seven_a100_ddp_mb, 3),
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
        "pair_state_enabled": str(model.use_pair_state if pair_operator_model else False),
        "pair_state_representation": _pair_state_representation(pair_operator_model, model.use_pair_state),
        "pair_rank": model.pair_rank if pair_operator_model else 0,
        "pair_top_k": model.pair_top_k if pair_operator_model else 0,
        "network_blocks": model.network_blocks if pair_operator_model else 0,
    }


def _pair_state_representation(pair_operator_model: bool, enabled: bool) -> str:
    if not pair_operator_model:
        return "none"
    if enabled:
        return "low_rank"
    return "disabled_low_rank_parameters_present"
