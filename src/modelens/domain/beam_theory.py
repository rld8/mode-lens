"""Euler-Bernoulli cantilever equations and restricted inverse calibration."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from modelens.domain.entities import BeamGeometry, Material, ModalMode, TwinFit
from modelens.domain.errors import NonIdentifiableError

FloatArray = npt.NDArray[np.float64]

CANTILEVER_BETA = np.array(
    [1.875104068711961, 4.694091132974174, 7.854757438237613, 10.995540734875467],
    dtype=np.float64,
)


def cantilever_mode_shape(position_normalized: FloatArray, mode_index: int) -> FloatArray:
    """Evaluate an exact Euler-Bernoulli cantilever eigenfunction."""
    if mode_index < 1 or mode_index > CANTILEVER_BETA.size:
        raise ValueError(f"mode_index must be between 1 and {CANTILEVER_BETA.size}")
    x = np.asarray(position_normalized, dtype=np.float64)
    if np.any((x < 0.0) | (x > 1.0)):
        raise ValueError("Normalized positions must lie in [0, 1]")
    beta = float(CANTILEVER_BETA[mode_index - 1])
    sigma = (np.cosh(beta) + np.cos(beta)) / (np.sinh(beta) + np.sin(beta))
    shape = np.cosh(beta * x) - np.cos(beta * x) - sigma * (np.sinh(beta * x) - np.sin(beta * x))
    maximum = float(np.max(np.abs(shape)))
    return np.asarray(shape / maximum, dtype=np.float64)


def cantilever_frequencies_hz(
    geometry: BeamGeometry,
    ei_over_rho_a_m4_s2: float,
    mode_count: int,
) -> FloatArray:
    """Calculate cantilever frequencies from the identifiable combined parameter EI/(rho*A)."""
    if ei_over_rho_a_m4_s2 <= 0.0:
        raise ValueError("EI/(rho*A) must be positive")
    if mode_count < 1 or mode_count > CANTILEVER_BETA.size:
        raise ValueError(f"mode_count must be between 1 and {CANTILEVER_BETA.size}")
    betas = CANTILEVER_BETA[:mode_count]
    values = betas**2 / (2.0 * np.pi * geometry.length_m**2) * np.sqrt(ei_over_rho_a_m4_s2)
    return np.asarray(values, dtype=np.float64)


def combined_parameter(geometry: BeamGeometry, material: Material) -> float:
    """Return EI/(rho*A) using known geometry, density and Young's modulus."""
    return (
        material.young_modulus_pa
        * geometry.second_moment_m4
        / (material.density_kg_m3 * geometry.area_m2)
    )


def calibrate_combined_parameter(
    measured_modes: tuple[ModalMode, ...],
    geometry: BeamGeometry,
    initial_value: float,
    monte_carlo_samples: int = 500,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> TwinFit:
    """Fit the only combined stiffness/mass parameter identifiable from frequencies alone."""
    count = min(len(measured_modes), CANTILEVER_BETA.size)
    if count == 0:
        raise NonIdentifiableError("At least one measured frequency is required")
    if initial_value <= 0.0:
        raise ValueError("Initial combined parameter must be positive")
    measured = np.array([mode.frequency_hz for mode in measured_modes[:count]], dtype=np.float64)
    standard_errors = np.array(
        [
            max(
                (mode.frequency_ci_hz[1] - mode.frequency_ci_hz[0]) / 3.92
                if mode.frequency_ci_hz is not None
                else 0.02 * mode.frequency_hz,
                1e-6,
            )
            for mode in measured_modes[:count]
        ],
        dtype=np.float64,
    )

    def residual(log_parameter: FloatArray) -> FloatArray:
        predicted = cantilever_frequencies_hz(geometry, float(np.exp(log_parameter[0])), count)
        return (predicted - measured) / standard_errors

    optimum = least_squares(residual, np.log(np.array([initial_value])), method="trf")
    fitted = float(np.exp(optimum.x[0]))
    predicted = cantilever_frequencies_hz(geometry, fitted, count)
    raw_residuals = predicted - measured
    interval: tuple[float, float] | None = None
    warnings = [
        "Frequencies alone identify EI/(rho*A), not E, I, rho and A separately.",
        "The Euler-Bernoulli model assumes a uniform slender beam and small deflection.",
    ]
    if monte_carlo_samples > 0:
        rng = np.random.default_rng(seed)
        sampled_frequencies = rng.normal(
            measured,
            standard_errors,
            size=(monte_carlo_samples, count),
        )
        sampled_frequencies = np.maximum(sampled_frequencies, np.finfo(np.float64).eps)
        if geometry.length_uncertainty_m > 0.0:
            sampled_lengths = rng.normal(
                geometry.length_m,
                geometry.length_uncertainty_m,
                size=monte_carlo_samples,
            )
            sampled_lengths = np.maximum(sampled_lengths, geometry.length_m * 0.1)
        else:
            sampled_lengths = np.full(monte_carlo_samples, geometry.length_m)
        coefficients = CANTILEVER_BETA[None, :count] ** 2 / (
            2.0 * np.pi * sampled_lengths[:, None] ** 2
        )
        weights = 1.0 / standard_errors**2
        fitted_sqrt = np.sum(coefficients * sampled_frequencies * weights, axis=1) / np.sum(
            coefficients**2 * weights,
            axis=1,
        )
        sampled_parameters = fitted_sqrt**2
        tail = (1.0 - confidence_level) / 2.0
        interval_values = np.quantile(sampled_parameters, [tail, 1.0 - tail])
        interval = float(interval_values[0]), float(interval_values[1])
        warnings.append(
            "The parameter interval is a seeded Monte Carlo propagation of frequency "
            "and free-length uncertainty."
        )
    else:
        warnings.append("Monte Carlo parameter uncertainty was disabled by configuration.")
    return TwinFit(
        parameter_name="EI/(rho*A) [m^4/s^2]",
        initial_value=initial_value,
        fitted_value=fitted,
        theoretical_frequencies_hz=predicted,
        residuals_hz=raw_residuals,
        weighted_cost=float(2.0 * optimum.cost),
        identifiable=True,
        confidence_interval=interval,
        warnings=tuple(warnings),
    )


def young_modulus_from_combined_parameter(
    ei_over_rho_a_m4_s2: float, geometry: BeamGeometry, density_kg_m3: float
) -> float:
    """Convert the combined parameter to E only when geometry and density are fixed."""
    if ei_over_rho_a_m4_s2 <= 0.0 or density_kg_m3 <= 0.0:
        raise ValueError("Combined parameter and density must be positive")
    return ei_over_rho_a_m4_s2 * density_kg_m3 * geometry.area_m2 / geometry.second_moment_m4
