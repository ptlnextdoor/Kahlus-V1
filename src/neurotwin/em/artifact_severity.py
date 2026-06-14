"""Artifact severity scoring + contamination map for Kahlus-EM Stage 0 (no-human, synthetic).

Pure numpy, deterministic. Quantifies how much synthetic environment/device artifact
contaminates idle/phantom EEG hardware recordings — channel-level, band-level, and overall —
with pass/warn/fail thresholds and a channel x frequency-band contamination map.

NO brain interpretation: in Stage 0 there is no brain, so all contamination is attributed to
sensor/environment artifact ``A_sensor(E_t) + noise``. Severity describes hardware/environment
data quality only; it is never a neuroscientific, causal, clinical, or consciousness claim.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from neurotwin.em.eeg_artifact_features import band_power, channel_artifact_features, compute_psd

#: Standard EEG frequency bands in Hz (``[low, high)``). A mains-line band is added per-call.
EEG_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

#: Overall severity thresholds (severity in [0, 1); higher = more contaminated).
SEVERITY_WARN_THRESHOLD: float = 0.2
SEVERITY_FAIL_THRESHOLD: float = 0.5

#: No-brain attribution note carried in every contamination map.
NO_BRAIN_ATTRIBUTION: str = "no_brain_synthetic_artifact_only: A_sensor(E_t) + noise"


def _squash(x: np.ndarray | float) -> np.ndarray | float:
    """Map a non-negative magnitude into [0, 1) monotonically: x / (1 + x). Finite-safe."""

    arr = np.asarray(x, dtype=np.float64)
    return arr / (1.0 + arr)


def _band_power_table(
    signal: np.ndarray,
    *,
    fs_hz: float,
    line_freq_hz: float,
    bands: Mapping[str, tuple[float, float]] = EEG_BANDS,
) -> dict[str, np.ndarray]:
    """Per-channel integrated power for each named band plus a mains-line band."""

    freqs, psd = compute_psd(signal, fs_hz)
    table: dict[str, np.ndarray] = {}
    for name, (low, high) in bands.items():
        table[name] = band_power(freqs, psd, low, high)
    table["line"] = band_power(freqs, psd, line_freq_hz - 2.0, line_freq_hz + 2.0)
    return table


def contamination_map(
    conditions: Mapping[str, np.ndarray],
    *,
    fs_hz: float,
    line_freq_hz: float,
    baseline_label: str = "baseline",
    bands: Mapping[str, tuple[float, float]] = EEG_BANDS,
) -> dict[str, Any]:
    """Build a channel x frequency-band contamination map across one or more conditions.

    ``conditions`` maps a condition label (e.g. ``"baseline"``, ``"perturbed_environment"``)
    to a ``(n_channels, n_samples)`` signal. The map records per-band power per condition and,
    for every non-baseline condition, the absolute per-band delta vs the baseline. Supports
    ``N >= 2`` idle/phantom/device states. JSON-serializable (numpy arrays -> lists).
    """

    if baseline_label not in conditions:
        raise ValueError(f"baseline_label {baseline_label!r} not present in conditions")

    band_names = list(bands.keys()) + ["line"]
    tables = {
        label: _band_power_table(sig, fs_hz=fs_hz, line_freq_hz=line_freq_hz, bands=bands)
        for label, sig in conditions.items()
    }
    n_channels = int(next(iter(conditions.values())).shape[0])

    conditions_out: dict[str, Any] = {}
    for label, table in tables.items():
        conditions_out[label] = {
            "band_power": {b: [float(v) for v in table[b]] for b in band_names},
            "band_power_mean": {b: float(np.mean(table[b])) for b in band_names},
        }

    base_table = tables[baseline_label]
    delta_out: dict[str, Any] = {}
    for label, table in tables.items():
        if label == baseline_label:
            continue
        delta_out[label] = {
            "band_power_abs_delta": {
                b: [float(v) for v in np.abs(table[b] - base_table[b])] for b in band_names
            },
            "band_power_mean_abs_delta": {
                b: float(np.mean(np.abs(table[b] - base_table[b]))) for b in band_names
            },
        }

    return {
        "bands": band_names,
        "n_channels": n_channels,
        "baseline_label": baseline_label,
        "conditions": conditions_out,
        "delta_vs_baseline": delta_out,
        "attribution": NO_BRAIN_ATTRIBUTION,
    }


def channel_contamination_score(
    baseline_signal: np.ndarray,
    condition_signal: np.ndarray,
    *,
    fs_hz: float,
    line_freq_hz: float,
) -> np.ndarray:
    """Per-channel contamination score in [0, 1): mains-line + relative broadband increase."""

    base = channel_artifact_features(baseline_signal, fs_hz, line_freq_hz)
    cond = channel_artifact_features(condition_signal, fs_hz, line_freq_hz)
    line_delta = np.abs(cond["line_noise_ratio"] - base["line_noise_ratio"])
    bb_base = np.where(base["broadband_power"] <= 0.0, 1.0, base["broadband_power"])
    bb_rel = np.abs(cond["broadband_power"] - base["broadband_power"]) / bb_base
    return np.asarray(_squash(line_delta + bb_rel), dtype=np.float64)


def band_contamination_score(cmap: Mapping[str, Any], condition_label: str) -> dict[str, float]:
    """Per-band contamination score in [0, 1) from the contamination map's mean abs deltas."""

    baseline_label = cmap["baseline_label"]
    base_mean = cmap["conditions"][baseline_label]["band_power_mean"]
    delta_mean = cmap["delta_vs_baseline"][condition_label]["band_power_mean_abs_delta"]
    scores: dict[str, float] = {}
    for band in cmap["bands"]:
        denom = base_mean[band] if base_mean[band] > 0.0 else 1.0
        scores[band] = float(_squash(delta_mean[band] / denom))
    return scores


def overall_artifact_severity(
    channel_scores: np.ndarray,
    band_scores: Mapping[str, float],
) -> float:
    """Overall severity in [0, 1): equal blend of mean channel and mean band contamination."""

    ch = float(np.mean(channel_scores)) if np.size(channel_scores) else 0.0
    bd = float(np.mean(list(band_scores.values()))) if band_scores else 0.0
    return 0.5 * ch + 0.5 * bd


def severity_verdict(overall: float) -> str:
    """Map an overall severity to ``pass`` (clean) / ``warn`` / ``fail`` (contaminated)."""

    if overall >= SEVERITY_FAIL_THRESHOLD:
        return "fail"
    if overall >= SEVERITY_WARN_THRESHOLD:
        return "warn"
    return "pass"


def artifact_severity_summary(
    baseline_signal: np.ndarray,
    condition_signal: np.ndarray,
    *,
    fs_hz: float,
    line_freq_hz: float,
    cmap: Mapping[str, Any],
    condition_label: str,
) -> dict[str, Any]:
    """Channel-, band-, and overall-level severity with a pass/warn/fail verdict."""

    channel_scores = channel_contamination_score(
        baseline_signal, condition_signal, fs_hz=fs_hz, line_freq_hz=line_freq_hz
    )
    band_scores = band_contamination_score(cmap, condition_label)
    overall = overall_artifact_severity(channel_scores, band_scores)
    finite = bool(
        np.isfinite(channel_scores).all()
        and all(np.isfinite(v) for v in band_scores.values())
        and np.isfinite(overall)
    )
    return {
        "channel_contamination_score": [float(v) for v in channel_scores],
        "channel_contamination_score_mean": float(np.mean(channel_scores)) if np.size(channel_scores) else 0.0,
        "band_contamination_score": band_scores,
        "overall_artifact_severity": float(overall),
        "verdict": severity_verdict(overall),
        "thresholds": {"warn": SEVERITY_WARN_THRESHOLD, "fail": SEVERITY_FAIL_THRESHOLD},
        "finite": finite,
    }
