"""Report export use case."""

from __future__ import annotations

from pathlib import Path

from modelens.domain.entities import AnalysisResult, ComparisonResult
from modelens.ports.reporting import ReportRenderer


class ExportReport:
    """Keep presentation technology outside the scientific workflow."""

    def __init__(self, renderer: ReportRenderer) -> None:
        """Store the replaceable report renderer."""
        self._renderer = renderer

    def analysis(self, result: AnalysisResult, destination: Path) -> Path:
        """Export a complete experiment report."""
        return self._renderer.render_analysis(result, destination)

    def comparison(self, result: ComparisonResult, destination: Path) -> Path:
        """Export a controlled-comparison report."""
        return self._renderer.render_comparison(result, destination)
