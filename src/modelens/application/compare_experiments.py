"""Experiment-comparison use case."""

from __future__ import annotations

from modelens.domain.comparison import compare_modes
from modelens.domain.entities import AnalysisResult, ComparisonResult


class CompareExperiments:
    """Compare measured results without implying a safety diagnosis."""

    def execute(self, baseline: AnalysisResult, modified: AnalysisResult) -> ComparisonResult:
        """Match modal sets by MAC and relative frequency."""
        return compare_modes(baseline.modal.modes, modified.modal.modes)
