from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from modelens.domain.entities import ModalAnalysis, SignalMatrix
from modelens.domain.modal_analysis import (
    estimate_damping_log_decrement,
    modal_assurance_criterion,
    normalize_mode_shape,
)


@given(st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False))
def test_mac_is_scale_and_sign_invariant(scale: float) -> None:
    shape = np.array([0.0, 0.2, -0.4, 1.0])
    assert modal_assurance_criterion(shape, -scale * shape) == pytest.approx(1.0)


def test_mac_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match="zero"):
        modal_assurance_criterion(np.zeros(3), np.ones(3))


def test_normalization_is_deterministic() -> None:
    normalized = normalize_mode_shape(np.array([0.0, -2.0, 1.0]))
    assert normalized[1] > 0.0
    assert np.linalg.norm(normalized) == pytest.approx(1.0)


def test_recovers_two_synthetic_frequencies(modal_result: ModalAnalysis) -> None:
    measured = [mode.frequency_hz for mode in modal_result.modes[:2]]
    assert measured[0] == pytest.approx(3.5, rel=0.02)
    assert measured[1] == pytest.approx(21.9, rel=0.02)


def test_damping_is_identified_for_clean_decay() -> None:
    fs = 120.0
    frequency = 4.0
    truth = 0.03
    time = np.arange(0.0, 8.0, 1.0 / fs)
    coordinate = np.exp(-truth * 2.0 * np.pi * frequency * time) * np.sin(
        2.0 * np.pi * frequency * time
    )
    estimate, flags = estimate_damping_log_decrement(coordinate, fs, frequency)
    assert not flags
    assert estimate == pytest.approx(truth, rel=0.15)


def test_preprocessed_signal_stays_finite(synthetic_signal: SignalMatrix) -> None:
    assert np.isfinite(synthetic_signal.cleaned_displacement).all()
    assert "linear_detrend" in synthetic_signal.transformations
