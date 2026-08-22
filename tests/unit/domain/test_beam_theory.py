from __future__ import annotations

import numpy as np
import pytest

from modelens.domain.beam_theory import (
    calibrate_combined_parameter,
    cantilever_frequencies_hz,
    cantilever_mode_shape,
    combined_parameter,
    young_modulus_from_combined_parameter,
)
from modelens.domain.entities import BeamGeometry, Material, ModalMode


def test_cantilever_mode_respects_clamped_boundary() -> None:
    position = np.linspace(0.0, 1.0, 20)
    shape = cantilever_mode_shape(position, 1)
    assert shape[0] == pytest.approx(0.0, abs=1e-12)
    assert np.max(np.abs(shape)) == pytest.approx(1.0)


def test_frequency_and_parameter_round_trip() -> None:
    geometry = BeamGeometry(0.30, 0.025, 0.003)
    material = Material("fixture", 500e6, 1180.0)
    parameter = combined_parameter(geometry, material)
    frequencies = cantilever_frequencies_hz(geometry, parameter, 3)
    modes = tuple(
        ModalMode(
            index + 1,
            value,
            0.02,
            cantilever_mode_shape(np.linspace(0, 1, 20), index + 1),
            1 / 3,
            1.0,
            (value * 0.99, value * 1.01),
        )
        for index, value in enumerate(frequencies)
    )
    fit = calibrate_combined_parameter(modes, geometry, parameter * 1.4)
    assert fit.fitted_value == pytest.approx(parameter, rel=1e-6)
    recovered_e = young_modulus_from_combined_parameter(fit.fitted_value, geometry, 1180.0)
    assert recovered_e == pytest.approx(material.young_modulus_pa, rel=1e-6)


@pytest.mark.parametrize("mode_count", [0, 5])
def test_rejects_unsupported_mode_count(mode_count: int) -> None:
    with pytest.raises(ValueError):
        cantilever_frequencies_hz(BeamGeometry(0.3, 0.02, 0.003), 0.3, mode_count)
