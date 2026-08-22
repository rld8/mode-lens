"""Standalone digital-twin calibration use case."""

from __future__ import annotations

from modelens.domain.beam_theory import calibrate_combined_parameter
from modelens.domain.entities import BeamGeometry, ModalMode, TwinFit


class CalibrateTwin:
    """Expose restricted calibration without coupling it to video I/O."""

    def execute(
        self,
        measured_modes: tuple[ModalMode, ...],
        geometry: BeamGeometry,
        initial_ei_over_rho_a: float,
        monte_carlo_samples: int = 500,
        confidence_level: float = 0.95,
        seed: int = 42,
    ) -> TwinFit:
        """Fit the identifiable combined parameter."""
        return calibrate_combined_parameter(
            measured_modes,
            geometry,
            initial_ei_over_rho_a,
            monte_carlo_samples,
            confidence_level,
            seed,
        )
