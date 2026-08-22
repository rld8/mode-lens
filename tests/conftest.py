from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

from modelens.config import ModalConfig, SignalConfig, UncertaintyConfig
from modelens.domain.beam_theory import cantilever_mode_shape
from modelens.domain.entities import (
    AnalysisResult,
    ModalAnalysis,
    QualityMetric,
    QualityReport,
    SignalMatrix,
    TrackingResult,
    VideoMetadata,
)
from modelens.domain.modal_analysis import identify_modes
from modelens.domain.signal_processing import preprocess_displacements


@pytest.fixture
def synthetic_signal() -> SignalMatrix:
    fs = 120.0
    time_s = np.arange(0.0, 8.0, 1.0 / fs)
    position = np.linspace(0.04, 0.98, 24)
    shape_1 = cantilever_mode_shape(position, 1)
    shape_2 = cantilever_mode_shape(position, 2)
    coordinate_1 = (
        1.2 * np.exp(-0.025 * 2 * np.pi * 3.5 * time_s) * np.sin(2 * np.pi * 3.5 * time_s)
    )
    coordinate_2 = (
        0.35 * np.exp(-0.015 * 2 * np.pi * 21.9 * time_s) * np.sin(2 * np.pi * 21.9 * time_s + 0.35)
    )
    rng = np.random.default_rng(42)
    raw = coordinate_1[:, None] * shape_1 + coordinate_2[:, None] * shape_2
    raw += rng.normal(0.0, 0.002, raw.shape)
    valid = np.ones(raw.shape, dtype=np.bool_)
    return preprocess_displacements(
        time_s,
        raw,
        valid,
        position,
        "mm",
        SignalConfig(highpass_hz=0.3, lowpass_hz=40.0),
    )


@pytest.fixture
def modal_result(synthetic_signal: SignalMatrix) -> ModalAnalysis:
    return identify_modes(
        synthetic_signal,
        ModalConfig(explained_energy=0.995, max_components=4, minimum_peak_prominence=0.04),
        UncertaintyConfig(bootstrap_samples=30, confidence_level=0.95),
        seed=42,
    )


@pytest.fixture
def analysis_result(synthetic_signal: SignalMatrix, modal_result: ModalAnalysis) -> AnalysisResult:
    frames, points = synthetic_signal.cleaned_displacement.shape
    reference = np.column_stack((np.linspace(10.0, 100.0, points), np.full(points, 50.0)))
    positions = np.repeat(reference[None, :, :], frames, axis=0)
    positions[:, :, 1] += synthetic_signal.raw_displacement
    tracking = TrackingResult(
        time_s=synthetic_signal.time_s,
        positions_px=positions,
        valid=synthetic_signal.valid,
        confidence=np.ones((frames, points)),
        reference_positions_px=reference,
        method="test",
    )
    return AnalysisResult(
        run_id="fixture-run",
        experiment_id=UUID("00000000-0000-0000-0000-000000000001"),
        created_at=datetime(2026, 8, 14, tzinfo=UTC),
        metadata=VideoMetadata(
            path=Path("fixture.mp4"),
            sha256="0" * 64,
            fps_nominal=120.0,
            fps_effective=120.0,
            frame_count=frames,
            duration_s=frames / 120.0,
            width_px=640,
            height_px=360,
            codec="mp4v",
        ),
        quality=QualityReport(
            score=100.0,
            metrics=(QualityMetric("contrast", 40.0, 20.0, True, "gray", "fixture"),),
        ),
        tracking=tracking,
        signal=synthetic_signal,
        modal=modal_result,
        twin=None,
        warnings=("fixture",),
        provenance={"seed": 42},
    )
