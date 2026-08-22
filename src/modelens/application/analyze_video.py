"""Application use case that orchestrates the complete video analysis pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from modelens import __version__
from modelens.config import AppConfig
from modelens.domain.beam_theory import calibrate_combined_parameter, combined_parameter
from modelens.domain.entities import AnalysisResult, BeamGeometry, Material
from modelens.domain.errors import InsufficientSignalError, InvalidExperimentError
from modelens.domain.modal_analysis import identify_modes
from modelens.domain.signal_processing import preprocess_displacements
from modelens.domain.units import pixels_to_metres
from modelens.ports.tracking import FeatureTracker, TrackingRequest
from modelens.ports.video import VideoSource


@dataclass(frozen=True, slots=True)
class AnalyzeVideoRequest:
    """Input path and validated settings for one analysis."""

    video_path: Path
    config: AppConfig


class AnalyzeVideo:
    """Coordinate I/O adapters and pure scientific services."""

    def __init__(self, video_source: VideoSource, tracker: FeatureTracker) -> None:
        """Store the two replaceable input adapters."""
        self._video_source = video_source
        self._tracker = tracker

    def execute(self, request: AnalyzeVideoRequest) -> AnalysisResult:
        """Run quality checks, tracking, signal analysis and optional twin calibration."""
        config = request.config
        experiment = config.experiment
        if experiment.axis_start_px is None or experiment.axis_end_px is None:
            raise InvalidExperimentError("Select or configure the beam support and free-end points")
        metadata = self._video_source.metadata(request.video_path, config.video)
        quality = self._video_source.quality(
            request.video_path, metadata, config.quality, experiment.roi
        )
        if not quality.can_analyze:
            raise InsufficientSignalError("; ".join(quality.blocking_reasons))
        tracking_request = TrackingRequest(
            axis_start_px=experiment.axis_start_px,
            axis_end_px=experiment.axis_end_px,
            roi=experiment.roi,
            settings=config.tracking,
        )
        tracking = self._tracker.track(
            self._video_source.frames(request.video_path), tracking_request
        )
        axis = np.asarray(experiment.axis_end_px) - np.asarray(experiment.axis_start_px)
        length_px = float(np.linalg.norm(axis))
        normal = np.array([-axis[1], axis[0]], dtype=np.float64) / length_px
        offsets = tracking.positions_px - tracking.reference_positions_px[None, :, :]
        displacement = np.einsum("tpc,c->tp", offsets, normal)
        displacement[~tracking.valid] = np.nan
        unit = "px"
        if experiment.meters_per_pixel is not None:
            displacement = pixels_to_metres(displacement, experiment.meters_per_pixel)
            unit = "m"
        spatial = (
            (tracking.reference_positions_px - np.asarray(experiment.axis_start_px))
            @ (axis / length_px)
            / length_px
        )
        matrix = preprocess_displacements(
            tracking.time_s,
            displacement,
            tracking.valid,
            np.asarray(spatial, dtype=np.float64),
            unit,
            config.signal,
        )
        modal = identify_modes(matrix, config.modal, config.uncertainty, config.project.seed)
        twin = None
        if experiment.geometry is not None and experiment.material is not None:
            geometry = BeamGeometry(**experiment.geometry.model_dump())
            material = Material(**experiment.material.model_dump())
            twin = calibrate_combined_parameter(
                modal.modes,
                geometry,
                combined_parameter(geometry, material),
                monte_carlo_samples=config.uncertainty.bootstrap_samples,
                confidence_level=config.uncertainty.confidence_level,
                seed=config.project.seed,
            )
        usable_maximum = config.video.usable_nyquist_ratio * metadata.fps_effective
        warnings = list(quality.warnings)
        if metadata.fps_effective < config.video.minimum_fps:
            warnings.append(
                f"Effective FPS {metadata.fps_effective:.2f} is below the recommended "
                f"{config.video.minimum_fps:.2f} FPS."
            )
        if any(mode.frequency_hz > usable_maximum for mode in modal.modes):
            warnings.append(
                "At least one peak exceeds the conservative observable band "
                f"({usable_maximum:.2f} Hz)."
            )
        warnings.append(
            "Educational experiment only: results are not a structural certification "
            "or safety diagnosis."
        )
        config_json = json.dumps(config.model_dump(mode="json"), sort_keys=True)
        run_id = hashlib.sha256(f"{metadata.sha256}:{config_json}".encode()).hexdigest()[:16]
        return AnalysisResult.now(
            run_id=run_id,
            experiment_id=uuid5(NAMESPACE_URL, f"modelens:{run_id}"),
            metadata=metadata,
            quality=quality,
            tracking=tracking,
            signal=matrix,
            modal=modal,
            twin=twin,
            warnings=tuple(warnings),
            provenance={
                "modelens_version": __version__,
                "video_sha256": metadata.sha256,
                "seed": config.project.seed,
                "config": config.model_dump(mode="json"),
            },
        )
