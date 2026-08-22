from __future__ import annotations

import numpy as np
import pytest

from modelens.application.calibrate_twin import CalibrateTwin
from modelens.domain.beam_theory import cantilever_frequencies_hz, cantilever_mode_shape
from modelens.domain.entities import BeamGeometry, ModalMode


def test_calibration_use_case_delegates_restricted_fit() -> None:
    geometry = BeamGeometry(0.3, 0.02, 0.003)
    truth = 0.31
    frequencies = cantilever_frequencies_hz(geometry, truth, 2)
    positions = np.linspace(0.0, 1.0, 12)
    modes = tuple(
        ModalMode(
            index + 1,
            frequency,
            0.02,
            cantilever_mode_shape(positions, index + 1),
            0.5,
            0.9,
            (frequency * 0.99, frequency * 1.01),
        )
        for index, frequency in enumerate(frequencies)
    )
    fit = CalibrateTwin().execute(modes, geometry, 0.5)
    assert fit.fitted_value == pytest.approx(truth, rel=1e-6)
