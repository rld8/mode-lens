"""Stable JSON representations at application boundaries."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np

from modelens.domain.entities import (
    AnalysisResult,
    BeamGeometry,
    ModalAnalysis,
    ModalMode,
    QualityMetric,
    QualityReport,
    SignalMatrix,
    TrackingResult,
    TwinFit,
    VideoMetadata,
)


def _array(value: Any, *, boolean: bool = False) -> np.ndarray:
    return np.asarray(value, dtype=np.bool_ if boolean else np.float64)


def analysis_result_to_dict(result: AnalysisResult, include_series: bool = True) -> dict[str, Any]:
    """Convert a result to a versioned, portable dictionary."""
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": result.run_id,
        "experiment_id": str(result.experiment_id),
        "created_at": result.created_at.isoformat(),
        "metadata": {
            "path": str(result.metadata.path),
            "sha256": result.metadata.sha256,
            "fps_nominal": result.metadata.fps_nominal,
            "fps_effective": result.metadata.fps_effective,
            "frame_count": result.metadata.frame_count,
            "duration_s": result.metadata.duration_s,
            "width_px": result.metadata.width_px,
            "height_px": result.metadata.height_px,
            "codec": result.metadata.codec,
        },
        "quality": {
            "score": result.quality.score,
            "metrics": [
                {
                    "name": metric.name,
                    "value": metric.value,
                    "threshold": metric.threshold,
                    "passed": metric.passed,
                    "unit": metric.unit,
                    "explanation": metric.explanation,
                }
                for metric in result.quality.metrics
            ],
            "warnings": list(result.quality.warnings),
            "blocking_reasons": list(result.quality.blocking_reasons),
        },
        "modal": {
            "modes": [
                {
                    "index": mode.index,
                    "frequency_hz": mode.frequency_hz,
                    "damping_ratio": mode.damping_ratio,
                    "shape": mode.shape.tolist(),
                    "energy_fraction": mode.energy_fraction,
                    "prominence": mode.prominence,
                    "frequency_ci_hz": list(mode.frequency_ci_hz)
                    if mode.frequency_ci_hz is not None
                    else None,
                    "damping_ci": list(mode.damping_ci) if mode.damping_ci is not None else None,
                    "quality_flags": list(mode.quality_flags),
                }
                for mode in result.modal.modes
            ],
            "singular_values": result.modal.singular_values.tolist(),
            "explained_energy": result.modal.explained_energy.tolist(),
        },
        "twin": None
        if result.twin is None
        else {
            "parameter_name": result.twin.parameter_name,
            "initial_value": result.twin.initial_value,
            "fitted_value": result.twin.fitted_value,
            "theoretical_frequencies_hz": result.twin.theoretical_frequencies_hz.tolist(),
            "residuals_hz": result.twin.residuals_hz.tolist(),
            "weighted_cost": result.twin.weighted_cost,
            "identifiable": result.twin.identifiable,
            "confidence_interval": list(result.twin.confidence_interval)
            if result.twin.confidence_interval is not None
            else None,
            "warnings": list(result.twin.warnings),
        },
        "warnings": list(result.warnings),
        "provenance": result.provenance,
    }
    if include_series:
        payload["tracking"] = {
            "time_s": result.tracking.time_s.tolist(),
            "positions_px": result.tracking.positions_px.tolist(),
            "valid": result.tracking.valid.tolist(),
            "confidence": result.tracking.confidence.tolist(),
            "reference_positions_px": result.tracking.reference_positions_px.tolist(),
            "method": result.tracking.method,
        }
        payload["signal"] = {
            "time_s": result.signal.time_s.tolist(),
            "position_normalized": result.signal.position_normalized.tolist(),
            "raw_displacement": result.signal.raw_displacement.tolist(),
            "cleaned_displacement": result.signal.cleaned_displacement.tolist(),
            "valid": result.signal.valid.tolist(),
            "sample_rate_hz": result.signal.sample_rate_hz,
            "unit": result.signal.unit,
            "transformations": list(result.signal.transformations),
        }
        payload["modal"].update(
            {
                "frequencies_hz": result.modal.frequencies_hz.tolist(),
                "component_psd": result.modal.component_psd.tolist(),
                "temporal_coordinates": result.modal.temporal_coordinates.tolist(),
            }
        )
    return payload


def analysis_result_from_dict(payload: dict[str, Any]) -> AnalysisResult:
    """Rebuild a domain result from the complete version-1 JSON representation."""
    if payload.get("schema_version") != "1.0":
        raise ValueError("Unsupported or missing result schema_version")
    if "tracking" not in payload or "signal" not in payload:
        raise ValueError("A summary-only result cannot be loaded as a complete analysis")
    metadata = payload["metadata"]
    quality = payload["quality"]
    tracking = payload["tracking"]
    signal_payload = payload["signal"]
    modal = payload["modal"]
    twin_payload = payload.get("twin")
    return AnalysisResult(
        run_id=str(payload["run_id"]),
        experiment_id=UUID(str(payload["experiment_id"])),
        created_at=datetime.fromisoformat(str(payload["created_at"])),
        metadata=VideoMetadata(
            path=Path(metadata["path"]),
            **{key: value for key, value in metadata.items() if key != "path"},
        ),
        quality=QualityReport(
            score=float(quality["score"]),
            metrics=tuple(QualityMetric(**metric) for metric in quality["metrics"]),
            warnings=tuple(quality["warnings"]),
            blocking_reasons=tuple(quality["blocking_reasons"]),
        ),
        tracking=TrackingResult(
            time_s=_array(tracking["time_s"]),
            positions_px=_array(tracking["positions_px"]),
            valid=_array(tracking["valid"], boolean=True),
            confidence=_array(tracking["confidence"]),
            reference_positions_px=_array(tracking["reference_positions_px"]),
            method=str(tracking["method"]),
        ),
        signal=SignalMatrix(
            time_s=_array(signal_payload["time_s"]),
            position_normalized=_array(signal_payload["position_normalized"]),
            raw_displacement=_array(signal_payload["raw_displacement"]),
            cleaned_displacement=_array(signal_payload["cleaned_displacement"]),
            valid=_array(signal_payload["valid"], boolean=True),
            sample_rate_hz=float(signal_payload["sample_rate_hz"]),
            unit=str(signal_payload["unit"]),
            transformations=tuple(signal_payload["transformations"]),
        ),
        modal=ModalAnalysis(
            modes=tuple(
                ModalMode(
                    index=int(mode["index"]),
                    frequency_hz=float(mode["frequency_hz"]),
                    damping_ratio=None
                    if mode["damping_ratio"] is None
                    else float(mode["damping_ratio"]),
                    shape=_array(mode["shape"]),
                    energy_fraction=float(mode["energy_fraction"]),
                    prominence=float(mode["prominence"]),
                    frequency_ci_hz=None
                    if mode["frequency_ci_hz"] is None
                    else tuple(mode["frequency_ci_hz"]),
                    damping_ci=None if mode["damping_ci"] is None else tuple(mode["damping_ci"]),
                    quality_flags=tuple(mode["quality_flags"]),
                )
                for mode in modal["modes"]
            ),
            singular_values=_array(modal["singular_values"]),
            explained_energy=_array(modal["explained_energy"]),
            frequencies_hz=_array(modal["frequencies_hz"]),
            component_psd=_array(modal["component_psd"]),
            temporal_coordinates=_array(modal["temporal_coordinates"]),
        ),
        twin=None
        if twin_payload is None
        else TwinFit(
            parameter_name=str(twin_payload["parameter_name"]),
            initial_value=float(twin_payload["initial_value"]),
            fitted_value=float(twin_payload["fitted_value"]),
            theoretical_frequencies_hz=_array(twin_payload["theoretical_frequencies_hz"]),
            residuals_hz=_array(twin_payload["residuals_hz"]),
            weighted_cost=float(twin_payload["weighted_cost"]),
            identifiable=bool(twin_payload["identifiable"]),
            confidence_interval=None
            if twin_payload["confidence_interval"] is None
            else tuple(twin_payload["confidence_interval"]),
            warnings=tuple(twin_payload["warnings"]),
        ),
        warnings=tuple(payload["warnings"]),
        provenance=dict(payload["provenance"]),
    )


def geometry_from_dict(payload: dict[str, Any]) -> BeamGeometry:
    """Build geometry from a JSON-compatible mapping."""
    return BeamGeometry(**payload)
