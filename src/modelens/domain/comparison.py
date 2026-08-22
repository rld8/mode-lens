"""Pair modal experiments using shape and frequency evidence."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from modelens.domain.entities import (
    ComparisonLabel,
    ComparisonResult,
    ModalMode,
    ModeMatch,
)
from modelens.domain.modal_analysis import modal_assurance_criterion


def _intervals_overlap(
    first: tuple[float, float] | None, second: tuple[float, float] | None
) -> bool | None:
    if first is None or second is None:
        return None
    return max(first[0], second[0]) <= min(first[1], second[1])


def compare_modes(
    baseline: tuple[ModalMode, ...],
    modified: tuple[ModalMode, ...],
    minimum_mac: float = 0.70,
    relative_change_threshold: float = 0.02,
) -> ComparisonResult:
    """Match two modal sets and label only evidence from this controlled experiment."""
    if not baseline or not modified:
        return ComparisonResult(
            matches=(),
            mac_matrix=np.empty((len(baseline), len(modified)), dtype=np.float64),
            unmatched_baseline=tuple(mode.index for mode in baseline),
            unmatched_modified=tuple(mode.index for mode in modified),
            warnings=("No comparison is possible without at least one mode in each experiment.",),
        )
    if not 0.0 <= minimum_mac <= 1.0 or relative_change_threshold < 0.0:
        raise ValueError("Comparison thresholds are outside their valid range")
    mac = np.array(
        [[modal_assurance_criterion(a.shape, b.shape) for b in modified] for a in baseline],
        dtype=np.float64,
    )
    frequency_penalty = np.array(
        [
            [abs(a.frequency_hz - b.frequency_hz) / a.frequency_hz for b in modified]
            for a in baseline
        ],
        dtype=np.float64,
    )
    rows, columns = linear_sum_assignment((1.0 - mac) + 0.25 * frequency_penalty)
    matches: list[ModeMatch] = []
    accepted_rows: set[int] = set()
    accepted_columns: set[int] = set()
    for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
        if mac[row, column] < minimum_mac:
            continue
        a = baseline[row]
        b = modified[column]
        relative_change = (b.frequency_hz - a.frequency_hz) / a.frequency_hz
        overlap = _intervals_overlap(a.frequency_ci_hz, b.frequency_ci_hz)
        if overlap is None:
            label = ComparisonLabel.INCONCLUSIVE
            reason = "Frequency uncertainty is unavailable for at least one mode."
        elif overlap and abs(relative_change) < relative_change_threshold:
            label = ComparisonLabel.STABLE
            reason = "Intervals overlap and the point change is below the practical threshold."
        elif overlap:
            label = ComparisonLabel.INCONCLUSIVE
            reason = "Point estimates differ, but their uncertainty intervals overlap."
        elif abs(relative_change) >= relative_change_threshold:
            label = ComparisonLabel.MEASURABLE_CHANGE
            reason = "Intervals do not overlap and the configured relative threshold is exceeded."
        else:
            label = ComparisonLabel.INCONCLUSIVE
            reason = "Intervals differ but the measured change is below the practical threshold."
        damping_change = (
            b.damping_ratio - a.damping_ratio
            if a.damping_ratio is not None and b.damping_ratio is not None
            else None
        )
        matches.append(
            ModeMatch(
                baseline_index=a.index,
                modified_index=b.index,
                mac=float(mac[row, column]),
                relative_frequency_change=float(relative_change),
                damping_change=damping_change,
                label=label,
                reason=reason,
            )
        )
        accepted_rows.add(row)
        accepted_columns.add(column)
    return ComparisonResult(
        matches=tuple(matches),
        mac_matrix=mac,
        unmatched_baseline=tuple(
            baseline[index].index for index in range(len(baseline)) if index not in accepted_rows
        ),
        unmatched_modified=tuple(
            modified[index].index for index in range(len(modified)) if index not in accepted_columns
        ),
        warnings=(
            "A measured change in this controlled test is not a structural safety diagnosis.",
            "MAC measures shape similarity; it does not establish damage or integrity.",
        ),
    )
