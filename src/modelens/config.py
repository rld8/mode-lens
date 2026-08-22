"""Typed configuration loaded from YAML and environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class FrozenModel(BaseModel):
    """Base class for immutable, strict application configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ProjectConfig(FrozenModel):
    """Run-level reproducibility settings."""

    seed: int = 42
    run_directory: Path = Path("runs")


class VideoConfig(FrozenModel):
    """Video acceptance limits."""

    max_duration_s: float = Field(30.0, gt=0.0)
    max_file_mb: float = Field(250.0, gt=0.0)
    minimum_fps: float = Field(30.0, gt=0.0)
    usable_nyquist_ratio: float = Field(0.40, gt=0.0, le=0.5)


class QualityConfig(FrozenModel):
    """Capture-quality warning thresholds."""

    blur_threshold: float = Field(80.0, ge=0.0)
    minimum_contrast: float = Field(20.0, ge=0.0)
    maximum_camera_motion_px: float = Field(2.5, ge=0.0)


class TrackingConfig(FrozenModel):
    """Feature tracking parameters."""

    method: Literal["lucas_kanade", "contour"] = "lucas_kanade"
    points: int = Field(32, ge=12, le=100)
    window_size: int = Field(21, ge=5, le=61)
    pyramid_levels: int = Field(3, ge=0, le=6)
    forward_backward_px: float = Field(1.5, gt=0.0)
    max_missing_ratio: float = Field(0.20, ge=0.0, lt=1.0)
    search_radius_px: int = Field(45, ge=5, le=300)

    @model_validator(mode="after")
    def odd_window(self) -> TrackingConfig:
        """Lucas-Kanade requires a symmetric odd-sized window."""
        if self.window_size % 2 == 0:
            raise ValueError("tracking.window_size must be odd")
        return self


class SignalConfig(FrozenModel):
    """Signal-cleaning parameters."""

    highpass_hz: float = Field(0.2, ge=0.0)
    lowpass_hz: float | None = Field(None, gt=0.0)
    filter_order: int = Field(4, ge=1, le=10)
    maximum_gap_ms: float = Field(100.0, ge=0.0)
    hampel_window: int = Field(7, ge=1, le=99)
    hampel_sigma: float = Field(3.5, gt=0.0)


class ModalConfig(FrozenModel):
    """POD and spectral identification parameters."""

    explained_energy: float = Field(0.95, gt=0.0, le=1.0)
    max_components: int = Field(8, ge=1, le=32)
    minimum_peak_prominence: float = Field(0.05, gt=0.0, le=1.0)
    minimum_relative_modal_score: float = Field(0.002, gt=0.0, le=1.0)
    minimum_frequency_hz: float = Field(0.5, ge=0.0)


class UncertaintyConfig(FrozenModel):
    """Bootstrap settings."""

    bootstrap_samples: int = Field(200, ge=0, le=10_000)
    confidence_level: float = Field(0.95, gt=0.5, lt=1.0)
    fps_relative_std: float = Field(0.0, ge=0.0, le=0.5)
    scale_relative_std: float = Field(0.0, ge=0.0, le=0.5)


class GeometryConfig(FrozenModel):
    """Measured rectangular beam geometry."""

    length_m: float = Field(gt=0.0)
    width_m: float = Field(gt=0.0)
    thickness_m: float = Field(gt=0.0)
    length_uncertainty_m: float = Field(0.0, ge=0.0)
    width_uncertainty_m: float = Field(0.0, ge=0.0)
    thickness_uncertainty_m: float = Field(0.0, ge=0.0)


class MaterialConfig(FrozenModel):
    """Known material properties used by the constrained twin."""

    name: str = Field(min_length=1)
    young_modulus_pa: float = Field(gt=0.0)
    density_kg_m3: float = Field(gt=0.0)


class ExperimentConfig(FrozenModel):
    """User measurements and image coordinates for one experiment."""

    boundary_condition: Literal["cantilever"] = "cantilever"
    roi: tuple[int, int, int, int] | None = None
    axis_start_px: tuple[float, float] | None = None
    axis_end_px: tuple[float, float] | None = None
    meters_per_pixel: float | None = Field(None, gt=0.0)
    geometry: GeometryConfig | None = None
    material: MaterialConfig | None = None

    @model_validator(mode="after")
    def complete_axis(self) -> ExperimentConfig:
        """Reject a half-specified beam axis."""
        if (self.axis_start_px is None) != (self.axis_end_px is None):
            raise ValueError("axis_start_px and axis_end_px must be provided together")
        if self.roi is not None and (self.roi[2] <= 0 or self.roi[3] <= 0):
            raise ValueError("ROI width and height must be positive")
        return self


class AppConfig(FrozenModel):
    """Complete immutable ModeLens configuration."""

    project: ProjectConfig = ProjectConfig()
    video: VideoConfig = VideoConfig()
    quality: QualityConfig = QualityConfig()
    tracking: TrackingConfig = TrackingConfig()
    signal: SignalConfig = SignalConfig()
    modal: ModalConfig = ModalConfig()
    uncertainty: UncertaintyConfig = UncertaintyConfig()
    experiment: ExperimentConfig = ExperimentConfig()


class RuntimeSettings(BaseSettings):
    """Small environment-only overrides that never contain experiment data."""

    model_config = SettingsConfigDict(env_prefix="MODELENS_", env_file=".env", extra="ignore")
    run_directory: Path | None = None
    log_level: str = "INFO"


def load_config(path: Path) -> AppConfig:
    """Load and validate a YAML configuration file."""
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError("The YAML root must be a mapping")
    config = AppConfig.model_validate(raw)
    runtime = RuntimeSettings()
    if runtime.run_directory is None:
        return config
    return config.model_copy(
        update={
            "project": config.project.model_copy(update={"run_directory": runtime.run_directory})
        }
    )
