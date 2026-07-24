"""Passive PCI complexity features: LZ, permutation entropy, multiscale entropy, spectral slope."""
from __future__ import annotations

import math

import numpy as np


def lempel_ziv_complexity(signal: np.ndarray) -> float:
    """Normalized LZ complexity on a median-binarized 1D signal."""
    x = np.asarray(signal, dtype=np.float64).ravel()
    if x.size < 4:
        return 0.0
    threshold = float(np.median(x))
    binary = (x >= threshold).astype(np.int8)
    n = binary.size
    complexity = 1
    prefix_len = 1
    pointer = 0
    while pointer + prefix_len < n:
        candidate = binary[pointer : pointer + prefix_len + 1]
        history = binary[: pointer + prefix_len]
        if _contains_subsequence(history, candidate):
            prefix_len += 1
        else:
            complexity += 1
            pointer += prefix_len
            prefix_len = 1
    return float(complexity * np.log2(n) / n)


def _contains_subsequence(haystack: np.ndarray, needle: np.ndarray) -> bool:
    if needle.size > haystack.size:
        return False
    for start in range(haystack.size - needle.size + 1):
        if np.array_equal(haystack[start : start + needle.size], needle):
            return True
    return False


def permutation_entropy(signal: np.ndarray, *, order: int = 3, delay: int = 1) -> float:
    """Normalized permutation entropy (Bandt & Pompe) on a 1D signal."""
    x = np.asarray(signal, dtype=np.float64).ravel()
    n_patterns = x.size - (order - 1) * delay
    if n_patterns <= 1:
        return 0.0
    counts = np.zeros(math.factorial(order), dtype=np.float64)
    for start in range(n_patterns):
        window = x[start : start + order * delay : delay]
        ranked = tuple(np.argsort(window, kind="mergesort"))
        idx = _pattern_to_index(ranked, order=order)
        counts[idx] += 1.0
    probs = counts[counts > 0] / float(n_patterns)
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(float(math.factorial(order)))
    if max_entropy <= 0.0:
        return 0.0
    return float(entropy / max_entropy)


def _pattern_to_index(pattern: tuple[int, ...], *, order: int) -> int:
    used = set()
    ranks = []
    for value in pattern:
        rank = sum(1 for other in pattern if other < value and other not in used)
        ranks.append(rank)
        used.add(value)
    index = 0
    factorial = math.factorial(order)
    for rank in ranks:
        factorial //= order
        index += rank * factorial
        order -= 1
    return int(index)


def multiscale_entropy(
    signal: np.ndarray,
    *,
    scale: int = 2,
    m: int = 2,
    tolerance_ratio: float = 0.2,
) -> float:
    """Coarse-grained sample entropy at one scale (SampEn-style, deterministic)."""
    x = np.asarray(signal, dtype=np.float64).ravel()
    if x.size < (m + 2) * scale:
        return 0.0
    coarse = _coarse_grain(x, scale=scale)
    tolerance = tolerance_ratio * float(np.std(coarse))
    if tolerance <= 1e-12:
        return 0.0
    return _sample_entropy(coarse, m=m, tolerance=tolerance)


def _coarse_grain(signal: np.ndarray, *, scale: int) -> np.ndarray:
    n = signal.size // scale
    trimmed = signal[: n * scale].reshape(n, scale)
    return trimmed.mean(axis=1)


def _sample_entropy(signal: np.ndarray, *, m: int, tolerance: float) -> float:
    templates_m = _embedding(signal, m)
    templates_m1 = _embedding(signal, m + 1)
    if templates_m.shape[0] < 2 or templates_m1.shape[0] < 2:
        return 0.0
    count_m = _match_count(templates_m, tolerance)
    count_m1 = _match_count(templates_m1, tolerance)
    if count_m <= 1 or count_m1 <= 1 or count_m1 >= count_m:
        return 0.0
    ratio = (count_m1 - 1.0) / (count_m - 1.0)
    if ratio <= 0.0:
        return 0.0
    return float(-np.log(ratio))


def _embedding(signal: np.ndarray, dimension: int) -> np.ndarray:
    n = signal.size - dimension + 1
    if n <= 0:
        return np.empty((0, dimension), dtype=np.float64)
    return np.stack([signal[idx : idx + dimension] for idx in range(n)], axis=0)


def _match_count(templates: np.ndarray, tolerance: float) -> float:
    count = 0.0
    n = templates.shape[0]
    for i in range(n - 1):
        diffs = np.max(np.abs(templates[i + 1 :] - templates[i]), axis=1)
        count += float(np.sum(diffs <= tolerance))
    return count


def spectral_slope(signal: np.ndarray, *, min_freq_bin: int = 2, max_freq_bin: int | None = None) -> float:
    """1/f aperiodic exponent from log-log PSD slope."""
    x = np.asarray(signal, dtype=np.float64).ravel()
    if x.size < 8:
        return 0.0
    spectrum = np.abs(np.fft.rfft(x - np.mean(x))) ** 2
    freqs = np.arange(spectrum.size, dtype=np.float64)
    stop = max_freq_bin if max_freq_bin is not None else max(min_freq_bin + 2, spectrum.size // 2)
    stop = min(stop, spectrum.size)
    if stop <= min_freq_bin + 1:
        return 0.0
    xs = np.log(freqs[min_freq_bin:stop] + 1e-8)
    ys = np.log(spectrum[min_freq_bin:stop] + 1e-8)
    slope = np.polyfit(xs, ys, deg=1)[0]
    return float(slope)


def complexity_block(windows: np.ndarray) -> np.ndarray:
    """Per-window LZ, permutation entropy, multiscale entropy across EEG channels."""
    x = np.asarray(windows, dtype=np.float32)
    rows = []
    for window in x:
        channel_feats = []
        for channel in range(window.shape[1]):
            signal = window[:, channel]
            channel_feats.extend(
                [
                    lempel_ziv_complexity(signal),
                    permutation_entropy(signal),
                    multiscale_entropy(signal),
                ]
            )
        rows.append(channel_feats)
    return np.asarray(rows, dtype=np.float32)


def spectral_slope_block(windows: np.ndarray) -> np.ndarray:
    """Per-window spectral slope averaged across channels."""
    x = np.asarray(windows, dtype=np.float32)
    slopes = [np.mean([spectral_slope(window[:, channel]) for channel in range(window.shape[1])]) for window in x]
    return np.asarray(slopes, dtype=np.float32)[:, None]
