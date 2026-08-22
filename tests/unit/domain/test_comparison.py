from __future__ import annotations

import numpy as np

from modelens.domain.comparison import compare_modes
from modelens.domain.entities import ComparisonLabel, ModalMode


def _mode(index: int, frequency: float, interval: tuple[float, float] | None) -> ModalMode:
    return ModalMode(
        index=index,
        frequency_hz=frequency,
        damping_ratio=0.02,
        shape=np.array([0.0, 0.2, 0.6, 1.0]),
        energy_fraction=0.8,
        prominence=0.9,
        frequency_ci_hz=interval,
    )


def test_comparison_labels_non_overlapping_change() -> None:
    result = compare_modes((_mode(1, 10.0, (9.9, 10.1)),), (_mode(1, 9.0, (8.9, 9.1)),))
    assert result.matches[0].label is ComparisonLabel.MEASURABLE_CHANGE
    assert result.matches[0].relative_frequency_change == -0.1


def test_comparison_is_inconclusive_without_intervals() -> None:
    result = compare_modes((_mode(1, 10.0, None),), (_mode(1, 9.0, None),))
    assert result.matches[0].label is ComparisonLabel.INCONCLUSIVE


def test_large_point_change_with_overlapping_intervals_is_inconclusive() -> None:
    result = compare_modes(
        (_mode(1, 10.0, (8.0, 12.0)),),
        (_mode(1, 8.5, (8.0, 9.0)),),
    )
    assert result.matches[0].label is ComparisonLabel.INCONCLUSIVE


def test_empty_comparison_does_not_invent_matches() -> None:
    result = compare_modes((), (_mode(1, 10.0, None),))
    assert not result.matches
    assert result.unmatched_modified == (1,)
