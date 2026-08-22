"""Explicit, checked unit conversions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def pixels_to_metres(
    displacement_px: npt.NDArray[np.float64], meters_per_pixel: float
) -> npt.NDArray[np.float64]:
    """Convert pixel displacement to metres using an in-plane calibration."""
    if meters_per_pixel <= 0.0 or not np.isfinite(meters_per_pixel):
        raise ValueError("meters_per_pixel must be positive and finite")
    return displacement_px * meters_per_pixel


def milliseconds_to_samples(milliseconds: float, sample_rate_hz: float) -> int:
    """Convert a gap duration to a conservative whole number of samples."""
    if milliseconds < 0.0 or sample_rate_hz <= 0.0:
        raise ValueError("Duration cannot be negative and sample rate must be positive")
    return max(0, round(milliseconds * sample_rate_hz / 1000.0))
