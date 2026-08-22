"""Domain entities and validated numerical results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


class BoundaryCondition(StrEnum):
    """Boundary conditions supported by the one-dimensional v1 twin."""

    CANTILEVER = "cantilever"


class ComparisonLabel(StrEnum):
    """Non-diagnostic labels allowed for controlled comparisons."""

    STABLE = "stable"
    MEASURABLE_CHANGE = "measurable_change"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Metadata read from the encoded video rather than its filename."""

    path: Path
    sha256: str
    fps_nominal: float
    fps_effective: float
    frame_count: int
    duration_s: float
    width_px: int
    height_px: int
    codec: str

    def __post_init__(self) -> None:
        if self.fps_nominal <= 0.0 or self.frame_count <= 0:
            raise ValueError("Video FPS and frame count must be positive")
        if self.width_px <= 0 or self.height_px <= 0 or self.duration_s <= 0.0:
            raise ValueError("Video dimensions and duration must be positive")


@dataclass(frozen=True, slots=True)
class BeamGeometry:
    """Rectangular beam measurements in SI units."""

    length_m: float
    width_m: float
    thickness_m: float
    length_uncertainty_m: float = 0.0
    width_uncertainty_m: float = 0.0
    thickness_uncertainty_m: float = 0.0

    def __post_init__(self) -> None:
        if min(self.length_m, self.width_m, self.thickness_m) <= 0.0:
            raise ValueError("Beam dimensions must be positive")
        if (
            min(
                self.length_uncertainty_m,
                self.width_uncertainty_m,
                self.thickness_uncertainty_m,
            )
            < 0.0
        ):
            raise ValueError("Measurement uncertainties cannot be negative")

    @property
    def area_m2(self) -> float:
        """Return rectangular cross-sectional area."""
        return self.width_m * self.thickness_m

    @property
    def second_moment_m4(self) -> float:
        """Return the second moment for bending across the thickness."""
        return self.width_m * self.thickness_m**3 / 12.0


@dataclass(frozen=True, slots=True)
class Material:
    """Known material values used only in a constrained calibration."""

    name: str
    young_modulus_pa: float
    density_kg_m3: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Material name cannot be empty")
        if self.young_modulus_pa <= 0.0 or self.density_kg_m3 <= 0.0:
            raise ValueError("Material properties must be positive")


@dataclass(frozen=True, slots=True)
class QualityMetric:
    """One measured capture-quality indicator."""

    name: str
    value: float
    threshold: float | None
    passed: bool
    unit: str
    explanation: str


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Capture checks; warnings are explicit and never folded into fake certainty."""

    score: float
    metrics: tuple[QualityMetric, ...]
    warnings: tuple[str, ...] = ()
    blocking_reasons: tuple[str, ...] = ()

    @property
    def can_analyze(self) -> bool:
        """Return whether no hard quality failure blocks analysis."""
        return not self.blocking_reasons


@dataclass(frozen=True, slots=True)
class TrackingResult:
    """Multipoint pixel trajectories with per-observation validity and confidence."""

    time_s: FloatArray
    positions_px: FloatArray
    valid: BoolArray
    confidence: FloatArray
    reference_positions_px: FloatArray
    method: str

    def __post_init__(self) -> None:
        frames, points, coordinates = self.positions_px.shape
        if coordinates != 2 or self.time_s.shape != (frames,):
            raise ValueError("Tracking arrays have inconsistent time dimensions")
        if self.valid.shape != (frames, points) or self.confidence.shape != (frames, points):
            raise ValueError("Tracking mask/confidence shapes are inconsistent")
        if self.reference_positions_px.shape != (points, 2):
            raise ValueError("Reference positions must have shape (points, 2)")
        if not np.isfinite(self.time_s).all():
            raise ValueError("Tracking timestamps must be finite")


@dataclass(frozen=True, slots=True)
class SignalMatrix:
    """Clean scalar displacement at multiple beam stations."""

    time_s: FloatArray
    position_normalized: FloatArray
    raw_displacement: FloatArray
    cleaned_displacement: FloatArray
    valid: BoolArray
    sample_rate_hz: float
    unit: str
    transformations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.raw_displacement.shape != self.cleaned_displacement.shape:
            raise ValueError("Raw and cleaned signals must share a shape")
        if self.raw_displacement.shape != self.valid.shape:
            raise ValueError("Signal validity mask must match the signal")
        if self.raw_displacement.shape[0] != self.time_s.size:
            raise ValueError("Signal time axis is inconsistent")
        if self.raw_displacement.shape[1] != self.position_normalized.size:
            raise ValueError("Signal spatial axis is inconsistent")
        if self.sample_rate_hz <= 0.0:
            raise ValueError("Sample rate must be positive")
        if not np.isfinite(self.cleaned_displacement).all():
            raise ValueError("Cleaned signals must be finite")


@dataclass(frozen=True, slots=True)
class ModalMode:
    """One measured modal estimate."""

    index: int
    frequency_hz: float
    damping_ratio: float | None
    shape: FloatArray
    energy_fraction: float
    prominence: float
    frequency_ci_hz: tuple[float, float] | None = None
    damping_ci: tuple[float, float] | None = None
    quality_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.index < 1 or self.frequency_hz <= 0.0:
            raise ValueError("Mode index and frequency must be positive")
        if not 0.0 <= self.energy_fraction <= 1.0:
            raise ValueError("Mode energy must be in [0, 1]")
        if self.damping_ratio is not None and not 0.0 <= self.damping_ratio < 1.0:
            raise ValueError("Damping ratio must be in [0, 1)")
        if not np.isfinite(self.shape).all():
            raise ValueError("Mode shape must be finite")


@dataclass(frozen=True, slots=True)
class ModalAnalysis:
    """POD decomposition and the modal estimates derived from it."""

    modes: tuple[ModalMode, ...]
    singular_values: FloatArray
    explained_energy: FloatArray
    frequencies_hz: FloatArray
    component_psd: FloatArray
    temporal_coordinates: FloatArray


@dataclass(frozen=True, slots=True)
class TwinFit:
    """Restricted Euler-Bernoulli calibration result."""

    parameter_name: str
    initial_value: float
    fitted_value: float
    theoretical_frequencies_hz: FloatArray
    residuals_hz: FloatArray
    weighted_cost: float
    identifiable: bool
    confidence_interval: tuple[float, float] | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModeMatch:
    """A baseline/modified mode pairing."""

    baseline_index: int
    modified_index: int
    mac: float
    relative_frequency_change: float
    damping_change: float | None
    label: ComparisonLabel
    reason: str


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Controlled experiment comparison without a structural diagnosis."""

    matches: tuple[ModeMatch, ...]
    mac_matrix: FloatArray
    unmatched_baseline: tuple[int, ...]
    unmatched_modified: tuple[int, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Complete result shared by CLI, UI, repository and report adapters."""

    run_id: str
    experiment_id: UUID
    created_at: datetime
    metadata: VideoMetadata
    quality: QualityReport
    tracking: TrackingResult
    signal: SignalMatrix
    modal: ModalAnalysis
    twin: TwinFit | None
    warnings: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def now(cls, **values: Any) -> AnalysisResult:
        """Build a result with a timezone-aware creation time."""
        return cls(created_at=datetime.now(UTC), **values)
