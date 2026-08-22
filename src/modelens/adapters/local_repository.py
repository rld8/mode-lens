"""Local, explicit export of derived analysis artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from modelens.application.dto import analysis_result_from_dict, analysis_result_to_dict
from modelens.domain.entities import AnalysisResult


class LocalExperimentRepository:
    """Persist results under a caller-selected run directory; raw video is never copied."""

    def save(self, result: AnalysisResult, destination: Path) -> Path:
        """Write JSON, compact arrays and CSV exports atomically where practical."""
        destination.mkdir(parents=True, exist_ok=True)
        result_path = destination / "result.json"
        temporary = destination / ".result.json.tmp"
        temporary.write_text(
            json.dumps(analysis_result_to_dict(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(result_path)
        np.savez_compressed(
            destination / "arrays.npz",
            time_s=result.signal.time_s,
            position_normalized=result.signal.position_normalized,
            raw_displacement=result.signal.raw_displacement,
            cleaned_displacement=result.signal.cleaned_displacement,
            valid=result.signal.valid,
            tracking_positions_px=result.tracking.positions_px,
            tracking_confidence=result.tracking.confidence,
        )
        signal_columns: dict[str, object] = {"time_s": result.signal.time_s}
        for point in range(result.signal.cleaned_displacement.shape[1]):
            signal_columns[f"raw_{point:02d}_{result.signal.unit}"] = (
                result.signal.raw_displacement[:, point]
            )
            signal_columns[f"clean_{point:02d}_{result.signal.unit}"] = (
                result.signal.cleaned_displacement[:, point]
            )
            signal_columns[f"valid_{point:02d}"] = result.signal.valid[:, point]
        pd.DataFrame(signal_columns).to_csv(destination / "signals.csv", index=False)
        trajectory_rows = []
        for frame_index, time_s in enumerate(result.tracking.time_s):
            for point_index in range(result.tracking.positions_px.shape[1]):
                trajectory_rows.append(
                    {
                        "time_s": time_s,
                        "point": point_index,
                        "x_px": result.tracking.positions_px[frame_index, point_index, 0],
                        "y_px": result.tracking.positions_px[frame_index, point_index, 1],
                        "valid": result.tracking.valid[frame_index, point_index],
                        "confidence": result.tracking.confidence[frame_index, point_index],
                    }
                )
        pd.DataFrame(trajectory_rows).to_csv(destination / "trajectories.csv", index=False)
        return result_path

    def load(self, result_file: Path) -> AnalysisResult:
        """Read and validate a complete ModeLens JSON result."""
        with result_file.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("Result JSON root must be an object")
        return analysis_result_from_dict(payload)
