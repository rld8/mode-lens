"""Video input contract."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from modelens.config import QualityConfig, VideoConfig
from modelens.domain.entities import QualityReport, VideoMetadata


@dataclass(frozen=True, slots=True)
class Frame:
    """Decoded BGR frame and its presentation timestamp."""

    index: int
    timestamp_s: float
    image_bgr: npt.NDArray[np.uint8]


class VideoSource(Protocol):
    """Read validated metadata, frames and capture-quality indicators."""

    def metadata(self, path: Path, limits: VideoConfig) -> VideoMetadata:
        """Read validated container metadata."""
        ...

    def frames(
        self, path: Path, start_s: float = 0.0, end_s: float | None = None
    ) -> Iterable[Frame]:
        """Yield decoded frames for the requested interval."""
        ...

    def quality(
        self,
        path: Path,
        metadata: VideoMetadata,
        thresholds: QualityConfig,
        roi: tuple[int, int, int, int] | None = None,
    ) -> QualityReport:
        """Measure capture quality without changing the source."""
        ...
