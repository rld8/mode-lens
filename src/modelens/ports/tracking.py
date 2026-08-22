"""Interchangeable multipoint tracking contract."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from modelens.config import TrackingConfig
from modelens.domain.entities import TrackingResult
from modelens.ports.video import Frame


@dataclass(frozen=True, slots=True)
class TrackingRequest:
    """Image coordinates and thresholds required by a tracker."""

    axis_start_px: tuple[float, float]
    axis_end_px: tuple[float, float]
    roi: tuple[int, int, int, int] | None
    settings: TrackingConfig


class FeatureTracker(Protocol):
    """Track beam stations without exposing OpenCV to the application layer."""

    def track(self, frames: Iterable[Frame], request: TrackingRequest) -> TrackingResult:
        """Track configured beam stations across decoded frames."""
        ...
