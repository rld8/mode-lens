"""Reproducible cantilever video generation with recorded ground truth."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from modelens.domain.beam_theory import cantilever_mode_shape


@dataclass(frozen=True, slots=True)
class SyntheticMode:
    """Ground-truth damped modal coordinate used to render a video."""

    frequency_hz: float
    damping_ratio: float
    amplitude_px: float
    phase_rad: float = 0.0


@dataclass(frozen=True, slots=True)
class SyntheticVideoSpec:
    """All parameters required to reproduce a synthetic capture."""

    fps: float = 120.0
    duration_s: float = 10.0
    width_px: int = 1280
    height_px: int = 720
    axis_start_px: tuple[int, int] = (144, 360)
    axis_end_px: tuple[int, int] = (1140, 360)
    beam_thickness_px: int = 18
    noise_std_gray: float = 1.5
    seed: int = 42
    modes: tuple[SyntheticMode, ...] = (
        SyntheticMode(3.5, 0.025, 48.0, 0.0),
        SyntheticMode(21.9, 0.015, 10.0, 0.35),
    )


def generate_cantilever_video(
    destination: Path,
    spec: SyntheticVideoSpec,
    ground_truth_path: Path | None = None,
) -> dict[str, object]:
    """Render a bright vibrating beam on a textured, stationary background."""
    if spec.fps < 15.0 or spec.duration_s <= 0.0 or not spec.modes:
        raise ValueError(
            "Synthetic video requires fps>=15, positive duration and at least one mode"
        )
    if any(mode.frequency_hz >= spec.fps / 2.0 for mode in spec.modes):
        raise ValueError("A synthetic mode cannot meet or exceed the Nyquist frequency")
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
        spec.fps,
        (spec.width_px, spec.height_px),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not create {destination}")
    rng = np.random.default_rng(spec.seed)
    start = np.asarray(spec.axis_start_px, dtype=np.float64)
    end = np.asarray(spec.axis_end_px, dtype=np.float64)
    stations = np.linspace(0.0, 1.0, 240)
    base_x = start[0] + stations * (end[0] - start[0])
    base_y = start[1] + stations * (end[1] - start[1])
    shapes = [cantilever_mode_shape(stations, index + 1) for index in range(len(spec.modes))]
    frame_count = round(spec.fps * spec.duration_s)
    try:
        for frame_index in range(frame_count):
            time_s = frame_index / spec.fps
            image = _background(spec.width_px, spec.height_px)
            displacement = np.zeros_like(stations)
            for mode, shape in zip(spec.modes, shapes, strict=True):
                angular = 2.0 * np.pi * mode.frequency_hz
                decay = np.exp(-mode.damping_ratio * angular * time_s)
                coordinate = mode.amplitude_px * decay * np.sin(angular * time_s + mode.phase_rad)
                displacement += coordinate * shape
            points = np.column_stack((base_x, base_y + displacement)).round().astype(np.int32)
            cv2.polylines(
                image,
                [points.reshape(-1, 1, 2)],
                isClosed=False,
                color=(235, 225, 80),
                thickness=spec.beam_thickness_px,
                lineType=cv2.LINE_AA,
            )
            clamp_half_height = max(24, 3 * spec.beam_thickness_px)
            cv2.rectangle(
                image,
                (
                    max(0, round(start[0] - 0.05 * (end[0] - start[0]))),
                    max(0, round(start[1] - clamp_half_height)),
                ),
                (
                    round(start[0] + 1),
                    min(spec.height_px - 1, round(start[1] + clamp_half_height)),
                ),
                (105, 110, 118),
                -1,
            )
            for station in np.linspace(0.04, 0.98, 32):
                marker = int(np.argmin(np.abs(stations - station)))
                cv2.circle(image, tuple(points[marker]), 2, (25, 30, 35), -1, cv2.LINE_AA)
            scale_start_x = round(start[0])
            scale_end_x = round(start[0] + (end[0] - start[0]) / 3.0)
            scale_y = spec.height_px - 35
            cv2.line(
                image,
                (scale_start_x, scale_y),
                (scale_end_x, scale_y),
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                "100 mm",
                ((scale_start_x + scale_end_x) // 2 - 28, spec.height_px - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (205, 205, 205),
                1,
                cv2.LINE_AA,
            )
            if spec.noise_std_gray > 0.0:
                noise = rng.normal(0.0, spec.noise_std_gray, image.shape).astype(np.int16)
                image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            writer.write(image)
    finally:
        writer.release()
    ground_truth: dict[str, object] = {
        "generator": "modelens.synthetic.generate_cantilever_video",
        "synthetic": True,
        "fps": spec.fps,
        "duration_s": spec.duration_s,
        "frame_count": frame_count,
        "axis_start_px": list(spec.axis_start_px),
        "axis_end_px": list(spec.axis_end_px),
        "modes": [asdict(mode) for mode in spec.modes],
        "seed": spec.seed,
        "limitations": [
            "The renderer follows ideal Euler-Bernoulli mode shapes.",
            "Compression and additive image noise are simulated; rolling shutter is not.",
        ],
    }
    if ground_truth_path is not None:
        ground_truth_path.parent.mkdir(parents=True, exist_ok=True)
        ground_truth_path.write_text(
            json.dumps(ground_truth, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return ground_truth


def _background(width: int, height: int) -> np.ndarray:
    image = np.full((height, width, 3), (24, 27, 32), dtype=np.uint8)
    for x in range(20, width, 40):
        cv2.line(image, (x, 0), (x, height - 1), (34, 38, 45), 1)
    for y in range(20, height, 40):
        cv2.line(image, (0, y), (width - 1, y), (34, 38, 45), 1)
    return image
