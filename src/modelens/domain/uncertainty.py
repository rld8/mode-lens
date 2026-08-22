"""Deterministic bootstrap uncertainty for spectral peak estimates."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import signal

FloatArray = npt.NDArray[np.float64]


def confidence_interval(values: FloatArray, confidence_level: float) -> tuple[float, float]:
    """Return a two-sided percentile interval for finite samples."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot calculate an interval from no finite values")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0.5 and 1")
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(finite, [tail, 1.0 - tail])
    return float(lower), float(upper)


def bootstrap_peak_frequency(
    values: FloatArray,
    sample_rate_hz: float,
    target_frequency_hz: float,
    samples: int,
    confidence_level: float,
    seed: int,
    fps_relative_std: float = 0.0,
    scale_relative_std: float = 0.0,
) -> tuple[float, float] | None:
    """Bootstrap Welch windows and locate the peak near a target frequency."""
    if samples <= 0:
        return None
    if values.ndim != 1 or values.size < 64 or sample_rate_hz <= 0.0:
        raise ValueError("A one-dimensional signal with at least 64 samples is required")
    window_size = min(values.size, max(64, round(2.5 * sample_rate_hz)))
    starts = np.arange(0, values.size - window_size + 1, max(1, window_size // 4))
    if starts.size < 2:
        return None
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    lower = max(0.0, target_frequency_hz * 0.70)
    upper = min(sample_rate_hz / 2.0, target_frequency_hz * 1.30)
    nfft = max(2048, 2 ** int(np.ceil(np.log2(window_size))))
    for sample_index in range(samples):
        start = int(rng.choice(starts))
        fps_factor = max(0.1, float(rng.normal(1.0, fps_relative_std)))
        scale_factor = max(0.1, float(rng.normal(1.0, scale_relative_std)))
        segment = scale_factor * values[start : start + window_size]
        frequencies, power = signal.periodogram(
            segment,
            fs=sample_rate_hz * fps_factor,
            nfft=nfft,
        )
        band = (frequencies >= lower) & (frequencies <= upper)
        if not np.any(band):
            estimates[sample_index] = np.nan
            continue
        band_indices = np.flatnonzero(band)
        estimates[sample_index] = frequencies[band_indices[int(np.argmax(power[band]))]]
    finite = estimates[np.isfinite(estimates)]
    if finite.size < max(10, samples // 4):
        return None
    lower_ci, upper_ci = confidence_interval(finite, confidence_level)
    half_bin_hz = sample_rate_hz / nfft / 2.0
    return lower_ci - half_bin_hz, upper_ci + half_bin_hz
