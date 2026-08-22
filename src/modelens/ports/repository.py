"""Analysis persistence contract."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from modelens.domain.entities import AnalysisResult


class ExperimentRepository(Protocol):
    """Persist and load derived experiment results."""

    def save(self, result: AnalysisResult, destination: Path) -> Path:
        """Persist a complete derived result."""
        ...

    def load(self, result_file: Path) -> AnalysisResult:
        """Load a complete derived result."""
        ...
