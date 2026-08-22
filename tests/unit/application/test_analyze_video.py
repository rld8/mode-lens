from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

from modelens.application.analyze_video import AnalyzeVideo, AnalyzeVideoRequest
from modelens.config import (
    AppConfig,
    ExperimentConfig,
    ModalConfig,
    TrackingConfig,
    UncertaintyConfig,
)
from modelens.domain.beam_theory import cantilever_mode_shape
from modelens.domain.entities import QualityReport, TrackingResult, VideoMetadata
from modelens.ports.tracking import TrackingRequest
from modelens.ports.video import Frame


class FakeVideoSource:
    def metadata(self, path: Path, limits: object) -> VideoMetadata:
        return VideoMetadata(path, "a" * 64, 120.0, 120.0, 960, 8.0, 640, 360, "fake")

    def frames(
        self, path: Path, start_s: float = 0.0, end_s: float | None = None
    ) -> Iterable[Frame]:
        yield Frame(0, 0.0, np.zeros((8, 8, 3), dtype=np.uint8))

    def quality(
        self, path: Path, metadata: VideoMetadata, thresholds: object, roi: object = None
    ) -> QualityReport:
        return QualityReport(100.0, ())


class FakeTracker:
    def track(self, frames: Iterable[Frame], request: TrackingRequest) -> TrackingResult:
        list(frames)
        fs = 120.0
        time = np.arange(0.0, 8.0, 1.0 / fs)
        spatial = np.linspace(0.04, 0.98, 20)
        reference = np.column_stack((100.0 * spatial, np.zeros_like(spatial)))
        coordinate = 10.0 * np.exp(-0.02 * 2 * np.pi * 3.5 * time) * np.sin(2 * np.pi * 3.5 * time)
        positions = np.repeat(reference[None], time.size, axis=0)
        positions[:, :, 1] += coordinate[:, None] * cantilever_mode_shape(spatial, 1)
        return TrackingResult(
            time,
            positions,
            np.ones((time.size, spatial.size), dtype=bool),
            np.ones((time.size, spatial.size)),
            reference,
            "fake",
        )


def test_application_orchestrates_without_opencv_dependency() -> None:
    config = AppConfig(
        tracking=TrackingConfig(points=20),
        modal=ModalConfig(explained_energy=0.99, max_components=2),
        uncertainty=UncertaintyConfig(bootstrap_samples=10),
        experiment=ExperimentConfig(axis_start_px=(0.0, 0.0), axis_end_px=(100.0, 0.0)),
    )
    result = AnalyzeVideo(FakeVideoSource(), FakeTracker()).execute(
        AnalyzeVideoRequest(Path("fake.mp4"), config)
    )
    assert result.modal.modes[0].frequency_hz == pytest.approx(3.5, rel=0.03)
    assert result.provenance["video_sha256"] == "a" * 64
    assert result.run_id
