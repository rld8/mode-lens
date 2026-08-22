"""Report rendering contract."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from modelens.domain.entities import AnalysisResult, ComparisonResult


class ReportRenderer(Protocol):
    """Render analysis and comparison reports to a user-controlled destination."""

    def render_analysis(self, result: AnalysisResult, destination: Path) -> Path:
        """Render an analysis report."""
        ...

    def render_comparison(self, result: ComparisonResult, destination: Path) -> Path:
        """Render a controlled comparison report."""
        ...
