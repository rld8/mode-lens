from __future__ import annotations

import numpy as np
import pytest

from modelens.domain.uncertainty import bootstrap_peak_frequency, confidence_interval


def test_percentile_interval() -> None:
    lower, upper = confidence_interval(np.arange(101, dtype=float), 0.90)
    assert lower == pytest.approx(5.0)
    assert upper == pytest.approx(95.0)


def test_bootstrap_is_seeded_and_contains_truth() -> None:
    fs = 100.0
    time = np.arange(0.0, 8.0, 1.0 / fs)
    values = np.sin(2 * np.pi * 7.0 * time)
    first = bootstrap_peak_frequency(values, fs, 7.0, 30, 0.95, 123)
    second = bootstrap_peak_frequency(values, fs, 7.0, 30, 0.95, 123)
    assert first == second
    assert first is not None
    assert first[0] <= 7.0 <= first[1]


def test_zero_samples_returns_no_interval() -> None:
    assert bootstrap_peak_frequency(np.ones(64), 100.0, 5.0, 0, 0.95, 1) is None
