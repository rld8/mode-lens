from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from modelens.domain.entities import (
    BeamGeometry,
    Material,
    ModalMode,
    SignalMatrix,
    TrackingResult,
    VideoMetadata,
)


def test_video_metadata_rejects_non_physical_values() -> None:
    with pytest.raises(ValueError, match="FPS"):
        VideoMetadata(Path("x"), "0" * 64, 0.0, 0.0, 1, 1.0, 10, 10, "x")
    with pytest.raises(ValueError, match="dimensions"):
        VideoMetadata(Path("x"), "0" * 64, 30.0, 30.0, 1, 0.0, 10, 10, "x")


def test_geometry_and_material_reject_invalid_measurements() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        BeamGeometry(0.0, 0.02, 0.003)
    with pytest.raises(ValueError, match="uncertainties"):
        BeamGeometry(0.3, 0.02, 0.003, length_uncertainty_m=-0.1)
    with pytest.raises(ValueError, match="name"):
        Material(" ", 1.0, 1.0)
    with pytest.raises(ValueError, match="properties"):
        Material("test", -1.0, 1.0)


def test_tracking_result_checks_every_axis() -> None:
    time = np.arange(3, dtype=float)
    positions = np.zeros((3, 2, 2))
    valid = np.ones((3, 2), dtype=bool)
    confidence = np.ones((3, 2))
    reference = np.zeros((2, 2))
    with pytest.raises(ValueError, match="time dimensions"):
        TrackingResult(time[:2], positions, valid, confidence, reference, "test")
    with pytest.raises(ValueError, match="mask"):
        TrackingResult(time, positions, valid[:, :1], confidence, reference, "test")
    with pytest.raises(ValueError, match="Reference"):
        TrackingResult(time, positions, valid, confidence, reference[:1], "test")
    time[1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        TrackingResult(time, positions, valid, confidence, reference, "test")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("clean_shape", "Raw and cleaned"),
        ("mask_shape", "validity mask"),
        ("time_shape", "time axis"),
        ("space_shape", "spatial axis"),
        ("sample_rate", "Sample rate"),
        ("non_finite", "finite"),
    ],
)
def test_signal_matrix_rejects_inconsistent_data(change: str, message: str) -> None:
    raw = np.zeros((4, 2))
    clean = raw.copy()
    valid = np.ones_like(raw, dtype=bool)
    time = np.arange(4, dtype=float)
    spatial = np.array([0.1, 0.9])
    sample_rate = 10.0
    if change == "clean_shape":
        clean = clean[:, :1]
    elif change == "mask_shape":
        valid = valid[:, :1]
    elif change == "time_shape":
        time = time[:3]
    elif change == "space_shape":
        spatial = spatial[:1]
    elif change == "sample_rate":
        sample_rate = 0.0
    else:
        clean[0, 0] = np.nan
    with pytest.raises(ValueError, match=message):
        SignalMatrix(time, spatial, raw, clean, valid, sample_rate, "px", ())


@pytest.mark.parametrize(
    "mode",
    [
        ModalMode,
    ],
)
def test_modal_mode_rejects_invalid_values(mode: type[ModalMode]) -> None:
    shape = np.ones(3)
    with pytest.raises(ValueError, match="index"):
        mode(0, 1.0, 0.02, shape, 0.5, 1.0)
    with pytest.raises(ValueError, match="energy"):
        mode(1, 1.0, 0.02, shape, 1.5, 1.0)
    with pytest.raises(ValueError, match="Damping"):
        mode(1, 1.0, 1.0, shape, 0.5, 1.0)
    with pytest.raises(ValueError, match="finite"):
        mode(1, 1.0, 0.02, np.array([0.0, np.nan]), 0.5, 1.0)
