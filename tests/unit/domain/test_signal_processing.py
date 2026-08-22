from __future__ import annotations

import numpy as np
import pytest

from modelens.config import SignalConfig
from modelens.domain.errors import InsufficientSignalError
from modelens.domain.signal_processing import (
    hampel_filter,
    interpolate_short_gaps,
    preprocess_displacements,
)
from modelens.domain.units import milliseconds_to_samples, pixels_to_metres


def test_interpolate_short_internal_gap() -> None:
    values = np.array([0.0, 1.0, np.nan, np.nan, 4.0])
    valid = np.array([True, True, False, False, True])
    result = interpolate_short_gaps(values, valid, max_gap=2)
    np.testing.assert_allclose(result, [0.0, 1.0, 2.0, 3.0, 4.0])


def test_does_not_fill_edge_or_long_gap() -> None:
    values = np.array([np.nan, 1.0, np.nan, np.nan, 4.0])
    valid = np.isfinite(values)
    result = interpolate_short_gaps(values, valid, max_gap=1)
    assert np.isnan(result[[0, 2, 3]]).all()


def test_hampel_replaces_isolated_outlier() -> None:
    values = np.array([1.0, 1.1, 0.9, 50.0, 1.0, 1.1, 0.9])
    filtered, replaced = hampel_filter(values, half_window=2, sigma=3.0)
    assert replaced[3]
    assert filtered[3] == pytest.approx(1.1)


def test_preprocess_rejects_invalid_filter_band() -> None:
    time = np.arange(64) / 20.0
    values = np.column_stack((np.sin(time), np.cos(time)))
    with pytest.raises(InsufficientSignalError, match="Invalid filter band"):
        preprocess_displacements(
            time,
            values,
            np.ones_like(values, dtype=bool),
            np.array([0.2, 0.8]),
            "px",
            SignalConfig(highpass_hz=9.5, lowpass_hz=9.0),
        )


def test_unit_conversions() -> None:
    np.testing.assert_allclose(pixels_to_metres(np.array([2.0, -3.0]), 0.001), [0.002, -0.003])
    assert milliseconds_to_samples(100.0, 120.0) == 12
    with pytest.raises(ValueError):
        pixels_to_metres(np.array([1.0]), 0.0)
