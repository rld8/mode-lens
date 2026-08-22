"""Sparse optical-flow and bright-contour tracking adapters."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from modelens.domain.entities import TrackingResult
from modelens.domain.errors import TrackingError
from modelens.ports.tracking import TrackingRequest
from modelens.ports.video import Frame


def _reference_points(request: TrackingRequest) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = np.asarray(request.axis_start_px, dtype=np.float64)
    end = np.asarray(request.axis_end_px, dtype=np.float64)
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length < 10.0:
        raise TrackingError("The selected beam axis is shorter than 10 pixels")
    tangent = axis / length
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    stations = np.linspace(0.04, 0.98, request.settings.points)
    points = start + stations[:, None] * axis
    return points, tangent, normal


def _validate_survival(result: TrackingResult, maximum_missing_ratio: float) -> TrackingResult:
    missing_by_point = 1.0 - np.mean(result.valid, axis=0)
    surviving = missing_by_point <= maximum_missing_ratio
    if int(surviving.sum()) < 12:
        raise TrackingError(
            f"Only {int(surviving.sum())} points meet the missing-data limit; "
            "at least 12 are required"
        )
    return TrackingResult(
        time_s=result.time_s,
        positions_px=result.positions_px[:, surviving],
        valid=result.valid[:, surviving],
        confidence=result.confidence[:, surviving],
        reference_positions_px=result.reference_positions_px[surviving],
        method=result.method,
    )


def _stabilized_grayscale(
    frames: Iterable[Frame], roi: tuple[int, int, int, int] | None
) -> Iterable[tuple[float, np.ndarray]]:
    """Yield frames registered to the first through a background affine RANSAC fit."""
    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise TrackingError("No decoded frames were supplied") from exc
    reference = cv2.cvtColor(first.image_bgr, cv2.COLOR_BGR2GRAY)
    feature_mask = np.full(reference.shape, 255, dtype=np.uint8)
    if roi is not None:
        x, y, width, height = roi
        feature_mask[y : y + height, x : x + width] = 0
    features = cv2.goodFeaturesToTrack(
        reference,
        maxCorners=250,
        qualityLevel=0.01,
        minDistance=8,
        mask=feature_mask,
    )
    yield first.timestamp_s, reference
    for frame in iterator:
        current = cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2GRAY)
        if features is None or len(features) < 8:
            yield frame.timestamp_s, current
            continue
        moved, status, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
            reference, current, features, None
        )
        if moved is None or status is None or int(status.sum()) < 8:
            yield frame.timestamp_s, current
            continue
        valid = status.ravel().astype(bool)
        transform, _ = cv2.estimateAffinePartial2D(
            moved[valid, 0],
            features[valid, 0],
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
        )
        if transform is None:
            yield frame.timestamp_s, current
            continue
        corrected = cv2.warpAffine(
            current,
            transform,
            (reference.shape[1], reference.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        yield frame.timestamp_s, corrected


class LucasKanadeTracker:
    """Pyramidal Lucas-Kanade with a forward-backward consistency check."""

    def track(self, frames: Iterable[Frame], request: TrackingRequest) -> TrackingResult:
        """Track fixed beam stations through all supplied frames."""
        reference, _, _ = _reference_points(request)
        stabilized = iter(_stabilized_grayscale(frames, request.roi))
        first_time, previous_gray = next(stabilized)
        times = [first_time]
        positions_rows = [reference.copy()]
        valid_rows = [np.ones(reference.shape[0], dtype=np.bool_)]
        confidence_rows = [np.ones(reference.shape[0], dtype=np.float64)]
        previous_points = reference.astype(np.float32).reshape(-1, 1, 2)
        active = np.ones(reference.shape[0], dtype=np.bool_)
        window = (request.settings.window_size, request.settings.window_size)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        )
        for timestamp_s, current_gray in stabilized:
            position_row = np.full((reference.shape[0], 2), np.nan, dtype=np.float64)
            valid_row = np.zeros(reference.shape[0], dtype=np.bool_)
            confidence_row = np.zeros(reference.shape[0], dtype=np.float64)
            forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
                previous_gray,
                current_gray,
                previous_points,
                None,
                winSize=window,
                maxLevel=request.settings.pyramid_levels,
                criteria=criteria,
            )
            if forward is None or forward_status is None:
                active[:] = False
                break
            backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
                current_gray,
                previous_gray,
                forward,
                None,
                winSize=window,
                maxLevel=request.settings.pyramid_levels,
                criteria=criteria,
            )
            if backward is None or backward_status is None:
                active[:] = False
                break
            fb_error = np.linalg.norm(backward[:, 0] - previous_points[:, 0], axis=1)
            frame_valid = (
                active
                & forward_status.ravel().astype(bool)
                & backward_status.ravel().astype(bool)
                & (fb_error <= request.settings.forward_backward_px)
            )
            position_row[frame_valid] = forward[frame_valid, 0]
            valid_row[:] = frame_valid
            confidence_row[frame_valid] = np.exp(
                -fb_error[frame_valid] / request.settings.forward_backward_px
            )
            times.append(timestamp_s)
            positions_rows.append(position_row)
            valid_rows.append(valid_row)
            confidence_rows.append(confidence_row)
            previous_points[frame_valid, 0] = forward[frame_valid, 0]
            active &= frame_valid
            previous_gray = current_gray
        if len(times) < 2:
            raise TrackingError("At least two decoded frames are required")
        result = TrackingResult(
            time_s=np.asarray(times, dtype=np.float64),
            positions_px=np.asarray(positions_rows, dtype=np.float64),
            valid=np.asarray(valid_rows, dtype=np.bool_),
            confidence=np.asarray(confidence_rows, dtype=np.float64),
            reference_positions_px=reference,
            method="lucas_kanade_forward_backward_affine_stabilized",
        )
        return _validate_survival(result, request.settings.max_missing_ratio)


class BrightContourTracker:
    """Track a bright slender specimen by sampling its largest connected contour."""

    def track(self, frames: Iterable[Frame], request: TrackingRequest) -> TrackingResult:
        """Estimate the centreline displacement normal to the selected initial axis."""
        reference, tangent, normal = _reference_points(request)
        times: list[float] = []
        positions_rows: list[np.ndarray] = []
        valid_rows: list[np.ndarray] = []
        confidence_rows: list[np.ndarray] = []
        offsets_normal = np.arange(
            -request.settings.search_radius_px,
            request.settings.search_radius_px + 1,
            dtype=np.float64,
        )
        offsets_tangent = np.arange(-3, 4, dtype=np.float64)
        for frame in frames:
            position_row = np.full((reference.shape[0], 2), np.nan, dtype=np.float64)
            valid_row = np.zeros(reference.shape[0], dtype=np.bool_)
            confidence_row = np.zeros(reference.shape[0], dtype=np.float64)
            gray = cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if request.roi is not None:
                x, y, width, height = request.roi
                roi_mask = np.zeros_like(binary)
                roi_mask[y : y + height, x : x + width] = 255
                binary = cv2.bitwise_and(binary, roi_mask)
            component = self._largest_component(binary)
            for point_index, point in enumerate(reference):
                normal_grid, tangent_grid = np.meshgrid(offsets_normal, offsets_tangent)
                coordinates = (
                    point + normal_grid[..., None] * normal + tangent_grid[..., None] * tangent
                )
                xs = np.rint(coordinates[..., 0]).astype(int)
                ys = np.rint(coordinates[..., 1]).astype(int)
                inside = (
                    (xs >= 0) & (xs < component.shape[1]) & (ys >= 0) & (ys < component.shape[0])
                )
                weights = np.zeros(xs.shape, dtype=np.float64)
                weights[inside] = component[ys[inside], xs[inside]] / 255.0
                profile = np.mean(weights, axis=0)
                total = float(profile.sum())
                if total < 1.0:
                    continue
                displacement = float(np.dot(profile, offsets_normal) / total)
                position_row[point_index] = point + displacement * normal
                valid_row[point_index] = True
                confidence_row[point_index] = min(1.0, total / 8.0)
            times.append(frame.timestamp_s)
            positions_rows.append(position_row)
            valid_rows.append(valid_row)
            confidence_rows.append(confidence_row)
        if len(times) < 2:
            raise TrackingError("At least two decoded frames are required")
        result = TrackingResult(
            time_s=np.asarray(times, dtype=np.float64),
            positions_px=np.asarray(positions_rows, dtype=np.float64),
            valid=np.asarray(valid_rows, dtype=np.bool_),
            confidence=np.asarray(confidence_rows, dtype=np.float64),
            reference_positions_px=reference,
            method="largest_bright_contour",
        )
        return _validate_survival(result, request.settings.max_missing_ratio)

    @staticmethod
    def _largest_component(binary: np.ndarray) -> np.ndarray:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if count <= 1:
            return np.zeros_like(binary)
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return np.where(labels == largest, 255, 0).astype(np.uint8)


def render_tracking_overlay(
    input_path: str,
    output_path: str,
    result: TrackingResult,
    codec: str = "mp4v",
) -> None:
    """Write a compact visual audit of accepted and rejected tracked points."""
    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise TrackingError(f"Could not reopen source video for overlay: {input_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*codec),  # type: ignore[attr-defined]
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise TrackingError(f"Could not create overlay video: {output_path}")
    frame_index = 0
    try:
        while frame_index < result.time_s.size:
            ok, frame = capture.read()
            if not ok:
                break
            for point, is_valid, score in zip(
                result.positions_px[frame_index],
                result.valid[frame_index],
                result.confidence[frame_index],
                strict=True,
            ):
                if not np.isfinite(point).all():
                    continue
                color = (
                    (30, int(80 + 175 * score), 255 - int(175 * score)) if is_valid else (0, 0, 255)
                )
                cv2.circle(frame, tuple(np.rint(point).astype(int)), 3, color, -1, cv2.LINE_AA)
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()
