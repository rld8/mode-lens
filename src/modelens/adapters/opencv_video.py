"""OpenCV implementation of video decoding and capture-quality checks."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from modelens.config import QualityConfig, VideoConfig
from modelens.domain.entities import QualityMetric, QualityReport, VideoMetadata
from modelens.domain.errors import InvalidVideoError
from modelens.ports.video import Frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _codec_name(value: float) -> str:
    code = int(value)
    return "".join(chr((code >> (8 * index)) & 0xFF) for index in range(4)).strip("\x00")


def _open_capture(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise InvalidVideoError(f"OpenCV could not decode the video: {path}")
    return capture


def _effective_fps(path: Path, nominal_fps: float, frame_count: int) -> float:
    """Estimate decoded timestamp cadence when the backend exposes monotonic timestamps."""
    capture = _open_capture(path)
    timestamps: list[float] = []
    try:
        for _ in range(min(frame_count, 240)):
            ok, _ = capture.read()
            if not ok:
                break
            timestamp_s = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if timestamp_s >= 0.0:
                timestamps.append(timestamp_s)
    finally:
        capture.release()
    if len(timestamps) < 10:
        return nominal_fps
    deltas = np.diff(timestamps)
    usable = deltas[np.isfinite(deltas) & (deltas > 1e-6)]
    if usable.size < 8:
        return nominal_fps
    estimate = 1.0 / float(np.median(usable))
    return estimate if 0.25 * nominal_fps <= estimate <= 4.0 * nominal_fps else nominal_fps


class OpenCVVideoSource:
    """Read local videos without uploading or retaining user media."""

    def metadata(self, path: Path, limits: VideoConfig) -> VideoMetadata:
        """Read container metadata and enforce hard resource limits."""
        if not path.is_file():
            raise InvalidVideoError(f"Video file does not exist: {path}")
        size_mb = path.stat().st_size / (1024.0 * 1024.0)
        if size_mb > limits.max_file_mb:
            raise InvalidVideoError(
                f"Video is {size_mb:.1f} MB; the configured limit is {limits.max_file_mb:.1f} MB"
            )
        capture = _open_capture(path)
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            codec = _codec_name(capture.get(cv2.CAP_PROP_FOURCC)) or "unknown"
        finally:
            capture.release()
        if not np.isfinite(fps) or fps <= 0.0 or frame_count <= 0:
            raise InvalidVideoError("The container does not expose usable FPS/frame metadata")
        if fps < 15.0:
            raise InvalidVideoError(f"Video FPS is {fps:.2f}; at least 15 FPS is required")
        duration_s = frame_count / fps
        if duration_s <= 0.0 or duration_s > limits.max_duration_s:
            raise InvalidVideoError(
                f"Video duration is {duration_s:.2f} s; the configured maximum is "
                f"{limits.max_duration_s:.2f} s"
            )
        if width < 64 or height < 64 or width > 7680 or height > 4320:
            raise InvalidVideoError(f"Unsupported video dimensions: {width}x{height}")
        return VideoMetadata(
            path=path,
            sha256=_sha256(path),
            fps_nominal=fps,
            fps_effective=_effective_fps(path, fps, frame_count),
            frame_count=frame_count,
            duration_s=duration_s,
            width_px=width,
            height_px=height,
            codec=codec,
        )

    def frames(
        self, path: Path, start_s: float = 0.0, end_s: float | None = None
    ) -> Iterable[Frame]:
        """Yield decoded frames in presentation order."""
        if start_s < 0.0 or (end_s is not None and end_s <= start_s):
            raise ValueError("Invalid video time interval")
        capture = _open_capture(path)
        capture.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        index = round(start_s * fps)
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                container_time = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                timestamp = container_time if container_time > 0.0 else index / fps
                if end_s is not None and timestamp > end_s:
                    break
                yield Frame(
                    index=index,
                    timestamp_s=timestamp,
                    image_bgr=np.asarray(image, dtype=np.uint8),
                )
                index += 1
        finally:
            capture.release()

    def quality(
        self,
        path: Path,
        metadata: VideoMetadata,
        thresholds: QualityConfig,
        roi: tuple[int, int, int, int] | None = None,
    ) -> QualityReport:
        """Measure blur, contrast, saturation and apparent global camera motion."""
        sampled = self._sample_frames(path, metadata.frame_count, count=24)
        if not sampled:
            raise InvalidVideoError("No frames could be decoded for quality analysis")
        full_grayscale = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in sampled]
        grayscale = full_grayscale
        if roi is not None:
            x, y, width, height = roi
            grayscale = [frame[y : y + height, x : x + width] for frame in grayscale]
        blur = float(np.median([cv2.Laplacian(frame, cv2.CV_64F).var() for frame in grayscale]))
        contrast = float(np.median([frame.std() for frame in grayscale]))
        saturation = float(np.mean([np.mean((frame <= 2) | (frame >= 253)) for frame in grayscale]))
        camera_motion = self._camera_motion(full_grayscale)
        metrics = (
            QualityMetric(
                "blur_variance",
                blur,
                thresholds.blur_threshold,
                blur >= thresholds.blur_threshold,
                "Laplacian variance",
                "Higher values generally indicate sharper edges; scene content affects the scale.",
            ),
            QualityMetric(
                "contrast_std",
                contrast,
                thresholds.minimum_contrast,
                contrast >= thresholds.minimum_contrast,
                "gray levels",
                "Low contrast weakens optical-flow and contour observations.",
            ),
            QualityMetric(
                "saturated_fraction",
                saturation,
                0.20,
                saturation <= 0.20,
                "fraction",
                "Large clipped regions remove image gradients needed for tracking.",
            ),
            QualityMetric(
                "camera_motion",
                camera_motion,
                thresholds.maximum_camera_motion_px,
                camera_motion <= thresholds.maximum_camera_motion_px,
                "px/sample",
                "Median background-feature displacement between sampled frames.",
            ),
        )
        warnings = tuple(
            f"Capture check failed: {metric.name}={metric.value:.3g} {metric.unit}"
            for metric in metrics
            if not metric.passed
        )
        score = 100.0 * sum(metric.passed for metric in metrics) / len(metrics)
        return QualityReport(score=score, metrics=metrics, warnings=warnings)

    @staticmethod
    def _sample_frames(path: Path, frame_count: int, count: int) -> list[np.ndarray]:
        capture = _open_capture(path)
        indices = np.linspace(0, max(0, frame_count - 1), min(count, frame_count), dtype=int)
        sampled: list[np.ndarray] = []
        try:
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, image = capture.read()
                if ok:
                    sampled.append(image)
        finally:
            capture.release()
        return sampled

    @staticmethod
    def _camera_motion(grayscale: list[np.ndarray]) -> float:
        motions: list[float] = []
        for first, second in pairwise(grayscale):
            points = cv2.goodFeaturesToTrack(
                first, maxCorners=120, qualityLevel=0.02, minDistance=8
            )
            if points is None or len(points) < 8:
                continue
            moved, status, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
                first, second, points, None
            )
            if moved is None or status is None:
                continue
            valid = status.ravel().astype(bool)
            if int(valid.sum()) < 8:
                continue
            displacement = moved[valid, 0] - points[valid, 0]
            median_vector = np.median(displacement, axis=0)
            motions.append(float(np.linalg.norm(median_vector)))
        return float(np.median(motions)) if motions else 0.0
