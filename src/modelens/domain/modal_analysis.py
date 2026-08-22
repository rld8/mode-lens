"""POD/SVD modal identification and explainable similarity metrics."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import signal

from modelens.config import ModalConfig, UncertaintyConfig
from modelens.domain.entities import ModalAnalysis, ModalMode, SignalMatrix
from modelens.domain.errors import InsufficientSignalError
from modelens.domain.uncertainty import bootstrap_peak_frequency

FloatArray = npt.NDArray[np.float64]


def normalize_mode_shape(shape: FloatArray) -> FloatArray:
    """Scale a mode to unit Euclidean norm with a deterministic sign."""
    vector = np.asarray(shape, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("A zero vector has no modal shape")
    normalized = vector / norm
    pivot = int(np.argmax(np.abs(normalized)))
    return -normalized if normalized[pivot] < 0.0 else normalized


def modal_assurance_criterion(shape_a: FloatArray, shape_b: FloatArray) -> float:
    """Return the Modal Assurance Criterion in the closed interval [0, 1]."""
    a = np.asarray(shape_a, dtype=np.float64).ravel()
    b = np.asarray(shape_b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("Mode shapes must have the same number of stations")
    denominator = float(np.dot(a, a) * np.dot(b, b))
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("MAC is undefined for a zero mode shape")
    value = float(abs(np.dot(a, b)) ** 2 / denominator)
    return float(np.clip(value, 0.0, 1.0))


def estimate_damping_log_decrement(
    coordinate: FloatArray, sample_rate_hz: float, frequency_hz: float
) -> tuple[float | None, tuple[str, ...]]:
    """Estimate damping from the exponential decay of positive modal peaks."""
    if coordinate.ndim != 1 or coordinate.size < 64:
        return None, ("damping_not_identifiable:too_few_samples",)
    low = max(0.05, 0.65 * frequency_hz)
    high = min(0.95 * sample_rate_hz / 2.0, 1.35 * frequency_hz)
    if low >= high:
        return None, ("damping_not_identifiable:invalid_band",)
    sos = signal.butter(3, [low, high], btype="bandpass", fs=sample_rate_hz, output="sos")
    try:
        filtered = signal.sosfiltfilt(sos, coordinate)
    except ValueError:
        return None, ("damping_not_identifiable:short_capture",)
    minimum_distance = max(1, int(0.70 * sample_rate_hz / frequency_hz))
    peaks, _ = signal.find_peaks(
        filtered,
        distance=minimum_distance,
        prominence=max(np.std(filtered) * 0.08, np.finfo(np.float64).eps),
    )
    if peaks.size < 5:
        return None, ("damping_not_identifiable:fewer_than_five_peaks",)
    amplitudes = filtered[peaks]
    keep = amplitudes > max(float(np.max(amplitudes)) * 0.08, np.finfo(np.float64).eps)
    peaks = peaks[keep]
    amplitudes = amplitudes[keep]
    if peaks.size < 5:
        return None, ("damping_not_identifiable:insufficient_decay_range",)
    times = peaks / sample_rate_hz
    slope, intercept = np.polyfit(times, np.log(amplitudes), 1)
    predicted = slope * times + intercept
    residual = float(np.sum((np.log(amplitudes) - predicted) ** 2))
    total = float(np.sum((np.log(amplitudes) - np.mean(np.log(amplitudes))) ** 2))
    r_squared = 1.0 - residual / total if total > 0.0 else 0.0
    if slope >= 0.0 or r_squared < 0.65:
        return None, (f"damping_not_identifiable:non_exponential_decay(r2={r_squared:.3f})",)
    decay_rate = -float(slope)
    angular_frequency = 2.0 * np.pi * frequency_hz
    damping = decay_rate / np.sqrt(angular_frequency**2 + decay_rate**2)
    if not 0.0 <= damping < 0.30:
        return None, ("damping_not_identifiable:implausible_estimate",)
    return float(damping), ()


def _spectral_mode_shape(
    centered: FloatArray, time_s: FloatArray, frequency_hz: float
) -> FloatArray:
    """Estimate the real spatial deflection pattern at one spectral frequency."""
    window = signal.windows.hann(centered.shape[0], sym=False)
    harmonic = np.exp(-2j * np.pi * frequency_hz * time_s)
    coefficients = np.sum(centered * (window * harmonic)[:, None], axis=0)
    pivot = int(np.argmax(np.abs(coefficients)))
    if abs(coefficients[pivot]) <= np.finfo(np.float64).eps:
        raise InsufficientSignalError("A candidate peak has no resolvable spatial shape")
    phase_aligned = coefficients * np.exp(-1j * np.angle(coefficients[pivot]))
    return normalize_mode_shape(np.asarray(np.real(phase_aligned), dtype=np.float64))


def _refine_peak_frequency(frequencies: FloatArray, power: FloatArray, index: int) -> float:
    """Refine an interior spectral-bin maximum by parabolic log-power interpolation."""
    if index <= 0 or index >= frequencies.size - 1:
        return float(frequencies[index])
    local = np.log(np.maximum(power[index - 1 : index + 2], np.finfo(np.float64).tiny))
    denominator = float(local[0] - 2.0 * local[1] + local[2])
    if abs(denominator) <= np.finfo(np.float64).eps:
        return float(frequencies[index])
    offset = float(np.clip(0.5 * (local[0] - local[2]) / denominator, -0.5, 0.5))
    return float(frequencies[index] + offset * (frequencies[index + 1] - frequencies[index]))


def identify_modes(
    matrix: SignalMatrix,
    settings: ModalConfig,
    uncertainty: UncertaintyConfig,
    seed: int,
) -> ModalAnalysis:
    """Identify dominant frequencies and spatial shapes from a cleaned signal matrix."""
    centered = matrix.cleaned_displacement - np.mean(matrix.cleaned_displacement, axis=0)
    if float(np.sqrt(np.mean(centered**2))) <= np.finfo(np.float64).eps * 100.0:
        raise InsufficientSignalError("The capture contains no measurable vibration")
    u_matrix, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    energy = singular_values**2
    energy_fraction = energy / float(np.sum(energy))
    cumulative = np.cumsum(energy_fraction)
    required = int(np.searchsorted(cumulative, settings.explained_energy) + 1)
    components = min(required, settings.max_components, singular_values.size)
    temporal = u_matrix[:, :components] * singular_values[:components]

    nperseg = min(centered.shape[0], max(128, round(4.0 * matrix.sample_rate_hz)))
    nfft = max(4096, 2 ** int(np.ceil(np.log2(nperseg))))
    frequencies, component_psd = signal.welch(
        temporal,
        fs=matrix.sample_rate_hz,
        axis=0,
        nperseg=nperseg,
        nfft=nfft,
        detrend="linear",
        scaling="density",
    )
    component_psd = component_psd.T
    candidates: list[tuple[float, int, float, float]] = []
    for component in range(components):
        power = component_psd[component]
        maximum = float(np.max(power))
        if maximum <= np.finfo(np.float64).eps:
            continue
        normalized = power / maximum
        valid_band = frequencies >= settings.minimum_frequency_hz
        peaks, properties = signal.find_peaks(
            normalized,
            prominence=settings.minimum_peak_prominence,
            distance=max(1, int(0.25 / max(frequencies[1] - frequencies[0], 1e-12))),
        )
        band_peaks = peaks[valid_band[peaks]]
        if band_peaks.size == 0:
            continue
        ordered_peaks = band_peaks[np.argsort(normalized[band_peaks])[::-1]][:3]
        for peak in ordered_peaks:
            if normalized[peak] < max(0.05, settings.minimum_peak_prominence):
                continue
            prominence_index = int(np.flatnonzero(peaks == peak)[0])
            prominence = float(properties["prominences"][prominence_index])
            candidates.append(
                (
                    _refine_peak_frequency(frequencies, power, int(peak)),
                    component,
                    prominence,
                    prominence * float(energy_fraction[component]),
                )
            )

    if not candidates:
        raise InsufficientSignalError("No spectral peak passed the configured prominence threshold")
    strongest_score = max(candidate[3] for candidate in candidates)
    candidates = [
        candidate
        for candidate in candidates
        if candidate[3] >= settings.minimum_relative_modal_score * strongest_score
    ]
    candidates.sort(key=lambda item: item[0])
    merged: list[tuple[float, int, float, float]] = []
    resolution = float(frequencies[1] - frequencies[0]) if frequencies.size > 1 else 0.0
    for candidate in candidates:
        if merged and abs(candidate[0] - merged[-1][0]) <= max(
            resolution * 1.5, 0.02 * candidate[0]
        ):
            if candidate[3] > merged[-1][3]:
                merged[-1] = candidate
        else:
            merged.append(candidate)
    merged = sorted(
        sorted(merged, key=lambda item: item[3], reverse=True)[: settings.max_components],
        key=lambda item: item[0],
    )

    modes: list[ModalMode] = []
    total_signal_energy = float(np.sum(centered**2))
    for mode_index, (frequency_hz, component, prominence, _) in enumerate(merged, start=1):
        mode_shape = _spectral_mode_shape(centered, matrix.time_s, frequency_hz)
        coordinate = np.asarray(centered @ mode_shape, dtype=np.float64)
        damping, flags = estimate_damping_log_decrement(
            coordinate, matrix.sample_rate_hz, frequency_hz
        )
        interval = bootstrap_peak_frequency(
            coordinate,
            matrix.sample_rate_hz,
            frequency_hz,
            uncertainty.bootstrap_samples,
            uncertainty.confidence_level,
            seed + component,
            uncertainty.fps_relative_std,
            uncertainty.scale_relative_std,
        )
        if interval is None:
            flags = (*flags, "frequency_uncertainty_not_identifiable")
        modes.append(
            ModalMode(
                index=mode_index,
                frequency_hz=frequency_hz,
                damping_ratio=damping,
                shape=mode_shape,
                energy_fraction=float(
                    np.clip(np.sum(coordinate**2) / total_signal_energy, 0.0, 1.0)
                ),
                prominence=prominence,
                frequency_ci_hz=interval,
                quality_flags=flags,
            )
        )
    return ModalAnalysis(
        modes=tuple(modes),
        singular_values=np.asarray(singular_values[:components], dtype=np.float64),
        explained_energy=np.asarray(energy_fraction[:components], dtype=np.float64),
        frequencies_hz=np.asarray(frequencies, dtype=np.float64),
        component_psd=np.asarray(component_psd, dtype=np.float64),
        temporal_coordinates=np.asarray(temporal, dtype=np.float64),
    )
