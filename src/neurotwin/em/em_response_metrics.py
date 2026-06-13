"""Descriptive EM-response metrics comparing artifact features across conditions.

Artifact model (documentation + code):

    Y_EEG_measured = Y_EEG_brain + A_sensor(E_t) + ε_t

In Stage 0 there is no brain (phantom/dummy), so ``Y_EEG_brain ≈ 0`` and any condition
difference is attributable to ``A_sensor(E_t)`` + noise. These metrics are descriptive
deltas only — they do not constitute a scientific or clinical claim.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

ARTIFACT_MODEL = "Y_EEG_measured = Y_EEG_brain + A_sensor(E_t) + eps_t"


def feature_delta(
    baseline: Mapping[str, np.ndarray],
    condition: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """Mean absolute per-channel delta for each shared feature key."""

    deltas: dict[str, float] = {}
    for key in sorted(set(baseline) & set(condition)):
        base = np.asarray(baseline[key], dtype=np.float64)
        cond = np.asarray(condition[key], dtype=np.float64)
        deltas[f"{key}_mean_abs_delta"] = float(np.mean(np.abs(cond - base)))
    return deltas


def summarize_em_response(
    baseline: Mapping[str, np.ndarray],
    condition: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Build a descriptive summary of how artifact features changed between conditions."""

    deltas = feature_delta(baseline, condition)
    finite = bool(all(np.isfinite(v) for v in deltas.values())) and bool(
        all(np.isfinite(np.asarray(v)).all() for v in baseline.values())
        and all(np.isfinite(np.asarray(v)).all() for v in condition.values())
    )
    detected = bool(any(v > 1e-9 for v in deltas.values()))
    return {
        "artifact_model": ARTIFACT_MODEL,
        "claim_status": "descriptive_no_human_artifact_audit",
        "feature_deltas": deltas,
        "finite": finite,
        "environment_effect_detected": detected,
    }
