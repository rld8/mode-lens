"""Pure signal preparation for multipoint vibration measurements."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import signal

from modelens.config import SignalConfig
from modelens.domain.entities import SignalMatrix
from modelens.domain.errors import InsufficientSignalError
from modelens.domain.units import milliseconds_to_samples

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def _contiguous_false_runs(valid: BoolArray) -> list[tuple[int, int]]:
    padded = np.pad(~valid, (1, 1), constant_values=False)
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def interpolate_short_gaps(values: FloatArray, valid: BoolArray, max_gap: int) -> FloatArray:
    """Linearly fill only internal invalid runs no longer than ``max_gap``."""
    if values.ndim != 1 or valid.shape != values.shape:
        raise ValueError("values and valid must be one-dimensional and aligned")
    result = values.astype(np.float64, copy=True)
    for start, end in _contiguous_false_runs(valid):
        is_internal = start > 0 and end < result.size
        if is_internal and end - start <= max_gap:
            result[start:end] = np.interp(
                np.arange(start, end),
                np.array([start - 1, end]),
                np.array([result[start - 1], result[end]]),
            )
    return result


def hampel_filter(
    values: FloatArray, half_window: int, sigma: float
) -> tuple[FloatArray, BoolArray]:
    """Replace isolated samples using a rolling median and robust MAD threshold."""
    if values.ndim != 1:
        raise ValueError("Hampel filter expects a one-dimensional signal")
    if half_window < 1 or sigma <= 0.0:
        raise ValueError("Hampel parameters must be positive")
    filtered = values.astype(np.float64, copy=True)
    replaced = np.zeros(values.size, dtype=np.bool_)
    for index in range(values.size):
        start = max(0, index - half_window)
        end = min(values.size, index + half_window + 1)
        window = values[start:end]
        median = float(np.median(window))
        mad = float(np.median(np.abs(window - median)))
        robust_scale = 1.4826 * mad
        if (
            robust_scale > np.finfo(np.float64).eps
            and abs(values[index] - median) > sigma * robust_scale
        ):
            filtered[index] = median
            replaced[index] = True
    return filtered, replaced


def _uniform_grid(
    time_s: FloatArray, values: FloatArray
) -> tuple[FloatArray, FloatArray, float, bool]:
    deltas = np.diff(time_s)
    if deltas.size < 2 or np.any(deltas <= 0.0):
        raise InsufficientSignalError("Timestamps must be strictly increasing")
    median_step = float(np.median(deltas))
    sample_rate_hz = 1.0 / median_step
    jitter_ratio = float(np.std(deltas) / median_step)
    if jitter_ratio <= 0.01:
        return time_s, values, sample_rate_hz, False
    grid = time_s[0] + np.arange(time_s.size, dtype=np.float64) * median_step
    resampled = np.column_stack(
        [np.interp(grid, time_s, values[:, column]) for column in range(values.shape[1])]
    )
    return grid, resampled, sample_rate_hz, True


def preprocess_displacements(
    time_s: FloatArray,
    displacement: FloatArray,
    valid: BoolArray,
    position_normalized: FloatArray,
    unit: str,
    settings: SignalConfig,
) -> SignalMatrix:
    """Interpolate short gaps, reject outliers, detrend and band-limit signals."""
    if displacement.ndim != 2 or valid.shape != displacement.shape:
        raise ValueError("Displacement and validity arrays must have shape (time, points)")
    if time_s.shape != (displacement.shape[0],):
        raise ValueError("Time axis is inconsistent with displacement")
    if position_normalized.shape != (displacement.shape[1],):
        raise ValueError("Spatial axis is inconsistent with displacement")
    if displacement.shape[0] < 32 or displacement.shape[1] < 2:
        raise InsufficientSignalError("At least 32 frames and two valid points are required")

    approximate_fs = 1.0 / float(np.median(np.diff(time_s)))
    max_gap = milliseconds_to_samples(settings.maximum_gap_ms, approximate_fs)
    filled = displacement.astype(np.float64, copy=True)
    original_valid = valid.copy()
    outlier_count = 0
    for column in range(filled.shape[1]):
        filled[:, column] = interpolate_short_gaps(filled[:, column], valid[:, column], max_gap)
        if not np.isfinite(filled[:, column]).all():
            raise InsufficientSignalError(
                f"Point {column} contains a gap longer than {settings.maximum_gap_ms:g} ms"
            )
        filled[:, column], outliers = hampel_filter(
            filled[:, column], settings.hampel_window, settings.hampel_sigma
        )
        outlier_count += int(outliers.sum())

    uniform_time, filled, sample_rate_hz, was_resampled = _uniform_grid(time_s, filled)
    detrended = signal.detrend(filled, axis=0, type="linear")
    nyquist_hz = sample_rate_hz / 2.0
    lowpass_hz = settings.lowpass_hz or 0.40 * sample_rate_hz
    lowpass_hz = min(lowpass_hz, 0.90 * nyquist_hz)
    if settings.highpass_hz >= lowpass_hz:
        raise InsufficientSignalError(
            f"Invalid filter band: {settings.highpass_hz:g}–{lowpass_hz:g} Hz"
        )
    sos = signal.butter(
        settings.filter_order,
        [settings.highpass_hz, lowpass_hz],
        btype="bandpass",
        fs=sample_rate_hz,
        output="sos",
    )
    try:
        cleaned = signal.sosfiltfilt(sos, detrended, axis=0)
    except ValueError as exc:
        raise InsufficientSignalError("The capture is too short for zero-phase filtering") from exc

    transformations = [
        f"interpolate_internal_gaps<={settings.maximum_gap_ms:g}ms",
        f"hampel(window={settings.hampel_window},sigma={settings.hampel_sigma:g},replaced={outlier_count})",
        "linear_detrend",
        f"butterworth_zero_phase(order={settings.filter_order},band={settings.highpass_hz:g}-{lowpass_hz:g}Hz)",
    ]
    if was_resampled:
        transformations.insert(0, "uniform_time_resample(jitter>1%)")
    return SignalMatrix(
        time_s=uniform_time,
        position_normalized=position_normalized,
        raw_displacement=displacement,
        cleaned_displacement=np.asarray(cleaned, dtype=np.float64),
        valid=original_valid,
        sample_rate_hz=sample_rate_hz,
        unit=unit,
        transformations=tuple(transformations),
    )
