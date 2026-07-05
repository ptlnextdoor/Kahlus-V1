"""PSD and per-channel artifact-feature extraction for EEG-like signals.

Pure numpy. Operates on ``(n_channels, n_samples)`` arrays. Used by the Stage 0 artifact
audit to summarize idle/phantom recordings; no brain interpretation is implied.
"""

from __future__ import annotations

import numpy as np


def _integrate(y: np.ndarray, x: np.ndarray, *, axis: int) -> np.ndarray:
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return trapezoid(y, x, axis=axis)


def compute_psd(signal: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(freqs, psd)`` for a ``(n_channels, n_samples)`` signal.

    ``psd`` has shape ``(n_channels, n_freqs)`` using a one-sided periodogram.
    """

    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2:
        raise ValueError("signal must be (n_channels, n_samples)")
    n_samples = signal.shape[1]
    if n_samples < 2:
        raise ValueError("signal must have at least two samples")
    detrended = signal - signal.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(detrended, axis=1)
    psd = (np.abs(spectrum) ** 2) / (fs_hz * n_samples)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs_hz)
    return freqs, psd


def band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> np.ndarray:
    """Per-channel integrated power in ``[low, high)`` Hz."""

    mask = (freqs >= low) & (freqs < high)
    if not mask.any():
        return np.zeros(psd.shape[0], dtype=np.float64)
    return _integrate(psd[:, mask], freqs[mask], axis=1)


def line_noise_ratio(freqs: np.ndarray, psd: np.ndarray, line_freq_hz: float, bandwidth_hz: float = 2.0) -> np.ndarray:
    """Per-channel ratio of mains-line band power to total power (artifact indicator)."""

    line = band_power(freqs, psd, line_freq_hz - bandwidth_hz, line_freq_hz + bandwidth_hz)
    total = _integrate(psd, freqs, axis=1)
    total = np.where(total <= 0.0, 1.0, total)
    return line / total


def _kurtosis(signal: np.ndarray) -> np.ndarray:
    centered = signal - signal.mean(axis=1, keepdims=True)
    var = np.mean(centered ** 2, axis=1)
    var = np.where(var <= 0.0, 1.0, var)
    return np.mean(centered ** 4, axis=1) / (var ** 2)


def channel_artifact_features(signal: np.ndarray, fs_hz: float, line_freq_hz: float = 60.0) -> dict[str, np.ndarray]:
    """Per-channel artifact features for an idle/phantom recording.

    Returns arrays of length ``n_channels`` for rms, broadband power, mains line-noise ratio,
    and excess-ish kurtosis. All finite for finite input.
    """

    signal = np.asarray(signal, dtype=np.float64)
    freqs, psd = compute_psd(signal, fs_hz)
    rms = np.sqrt(np.mean(signal ** 2, axis=1))
    broadband = _integrate(psd, freqs, axis=1)
    return {
        "rms": rms,
        "broadband_power": broadband,
        "line_noise_ratio": line_noise_ratio(freqs, psd, line_freq_hz),
        "kurtosis": _kurtosis(signal),
    }
