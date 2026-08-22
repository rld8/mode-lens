from __future__ import annotations

from pathlib import Path

import pytest

from modelens.adapters.opencv_tracking import LucasKanadeTracker, render_tracking_overlay
from modelens.adapters.opencv_video import OpenCVVideoSource
from modelens.application.analyze_video import AnalyzeVideoRequest
from modelens.bootstrap import create_analyzer
from modelens.config import ExperimentConfig, TrackingConfig, load_config
from modelens.ports.tracking import TrackingRequest
from modelens.synthetic import SyntheticMode, SyntheticVideoSpec, generate_cantilever_video


@pytest.mark.integration
def test_synthetic_video_recovers_known_frequency(tmp_path: Path) -> None:
    video = tmp_path / "single_mode.mp4"
    truth = 3.5
    spec = SyntheticVideoSpec(
        fps=60.0,
        duration_s=5.0,
        modes=(SyntheticMode(truth, 0.025, 48.0),),
    )
    generate_cantilever_video(
        video,
        spec,
    )
    root = Path(__file__).resolve().parents[2]
    base = load_config(root / "configs/demo_cantilever.yaml")
    config = base.model_copy(
        update={
            "tracking": TrackingConfig(method="contour", points=24, search_radius_px=70),
            "experiment": ExperimentConfig(
                roi=base.experiment.roi,
                axis_start_px=tuple(float(value) for value in spec.axis_start_px),
                axis_end_px=tuple(float(value) for value in spec.axis_end_px),
            ),
            "uncertainty": base.uncertainty.model_copy(update={"bootstrap_samples": 10}),
        }
    )
    result = create_analyzer(config).execute(AnalyzeVideoRequest(video, config))
    assert result.modal.modes[0].frequency_hz == pytest.approx(truth, rel=0.05)
    assert result.tracking.valid.mean() > 0.95


@pytest.mark.integration
def test_lucas_kanade_and_overlay_on_textured_demo(tmp_path: Path) -> None:
    video = tmp_path / "textured.mp4"
    spec = SyntheticVideoSpec(
        fps=60.0,
        duration_s=2.0,
        modes=(SyntheticMode(3.5, 0.02, 36.0),),
    )
    generate_cantilever_video(
        video,
        spec,
    )
    settings = TrackingConfig(
        method="lucas_kanade",
        points=32,
        forward_backward_px=3.0,
        max_missing_ratio=0.5,
    )
    request = TrackingRequest(
        tuple(float(value) for value in spec.axis_start_px),
        tuple(float(value) for value in spec.axis_end_px),
        None,
        settings,
    )
    tracking = LucasKanadeTracker().track(OpenCVVideoSource().frames(video), request)
    assert tracking.positions_px.shape[1] >= 12
    assert tracking.valid.mean() > 0.9
    overlay = tmp_path / "overlay.mp4"
    render_tracking_overlay(str(video), str(overlay), tracking)
    assert overlay.stat().st_size > 0
