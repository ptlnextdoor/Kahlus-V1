"""Stage 0 artifact audit: does environment/device change affect EEG hardware (no brain)?

Generates synthetic idle/phantom recordings under two EM contexts, extracts artifact
features, and produces a descriptive audit report. SAFETY: synthetic only, no human, no
stimulation. The "perturbed" condition adds a synthetic sensor-artifact term ``A_sensor(E_t)``
(mains line component + broadband), never a delivered field.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from neurotwin.em.eeg_artifact_features import channel_artifact_features
from neurotwin.em.em_context_schema import EMContext
from neurotwin.em.em_response_metrics import ARTIFACT_MODEL, summarize_em_response


def synthesize_idle_recording(
    *,
    seed: int,
    n_channels: int,
    n_samples: int,
    fs_hz: float,
    line_freq_hz: float = 60.0,
    em_field_strength_arb: float = 0.0,
) -> np.ndarray:
    """Synthesize a phantom/idle EEG-like recording ``(n_channels, n_samples)``.

    Baseline = white noise + faint mains line component. The synthetic sensor artifact
    ``A_sensor(E_t)`` scales with ``em_field_strength_arb`` and adds a stronger mains line
    component plus broadband pickup, modeling environmental/device coupling without a brain.
    """

    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs_hz
    base = rng.normal(scale=1.0, size=(n_channels, n_samples))
    faint_line = 0.05 * np.sin(2.0 * np.pi * line_freq_hz * t)[None, :]
    signal = base + faint_line
    if em_field_strength_arb > 0.0:
        phases = rng.uniform(0, 2 * np.pi, size=(n_channels, 1))
        a_sensor_line = em_field_strength_arb * np.sin(2.0 * np.pi * line_freq_hz * t[None, :] + phases)
        a_sensor_broadband = 0.3 * em_field_strength_arb * rng.normal(size=(n_channels, n_samples))
        signal = signal + a_sensor_line + a_sensor_broadband
    return signal


def run_artifact_audit(
    baseline_signal: np.ndarray,
    condition_signal: np.ndarray,
    *,
    fs_hz: float,
    line_freq_hz: float,
    baseline_context: EMContext,
    condition_context: EMContext,
) -> dict[str, Any]:
    """Compute artifact features for both conditions and summarize the descriptive response."""

    baseline_context.validate()
    condition_context.validate()
    baseline_feats = channel_artifact_features(baseline_signal, fs_hz, line_freq_hz)
    condition_feats = channel_artifact_features(condition_signal, fs_hz, line_freq_hz)
    response = summarize_em_response(baseline_feats, condition_feats)
    return {
        "schema": "kahlus.em_stage0_artifact_audit.v1",
        "branch": "em",
        "claim_status": "descriptive_no_human_artifact_audit",
        "artifact_model": ARTIFACT_MODEL,
        "fs_hz": float(fs_hz),
        "line_freq_hz": float(line_freq_hz),
        "baseline_context": baseline_context.to_dict(),
        "condition_context": condition_context.to_dict(),
        "baseline_features_mean": {k: float(np.mean(v)) for k, v in baseline_feats.items()},
        "condition_features_mean": {k: float(np.mean(v)) for k, v in condition_feats.items()},
        "response": response,
    }


def format_artifact_report_md(report: dict[str, Any]) -> str:
    response = report.get("response", {})
    lines = [
        "# Kahlus-EM Stage 0 Artifact Audit (no-human, synthetic)",
        "",
        f"- claim_status: {report.get('claim_status')}",
        f"- artifact_model: {report.get('artifact_model')}",
        f"- fs_hz: {report.get('fs_hz')}  line_freq_hz: {report.get('line_freq_hz')}",
        f"- environment_effect_detected: {response.get('environment_effect_detected')}",
        f"- finite: {response.get('finite')}",
        "",
        "## Feature deltas (baseline vs condition)",
        "",
    ]
    deltas = response.get("feature_deltas", {})
    if deltas:
        lines.extend(f"- {key}: {value:.6g}" for key, value in deltas.items())
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- No human subject. No stimulation. No high voltage. No clinical claim.",
            "- Synthetic phantom data only; this audit is descriptive infrastructure, not a result.",
            "",
        ]
    )
    return "\n".join(lines)
