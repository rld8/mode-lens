"""Serializable-enough session state helpers for the Streamlit interface."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from modelens.application.analyze_video import AnalyzeVideoRequest
from modelens.application.dto import analysis_result_to_dict
from modelens.bootstrap import create_analyzer
from modelens.config import AppConfig
from modelens.domain.entities import AnalysisResult, ComparisonResult

ANALYSIS_KEY = "modelens_analysis"
VIDEO_BYTES_KEY = "modelens_video_bytes"
VIDEO_NAME_KEY = "modelens_video_name"
COMPARISON_KEY = "modelens_comparison"


def repository_root() -> Path:
    """Return the source checkout root used by bundled demo assets."""
    return Path(__file__).resolve().parents[4]


def current_analysis() -> AnalysisResult | None:
    """Return the active analysis, if any."""
    value = st.session_state.get(ANALYSIS_KEY)
    return value if isinstance(value, AnalysisResult) else None


def current_comparison() -> ComparisonResult | None:
    """Return the active comparison, if any."""
    value = st.session_state.get(COMPARISON_KEY)
    return value if isinstance(value, ComparisonResult) else None


def analyse_path(path: Path, config: AppConfig, video_bytes: bytes | None = None) -> AnalysisResult:
    """Run analysis and retain only derived data plus optional in-memory video bytes."""
    result = create_analyzer(config).execute(AnalyzeVideoRequest(path, config))
    st.session_state[ANALYSIS_KEY] = result
    st.session_state[VIDEO_NAME_KEY] = path.name
    if video_bytes is not None:
        st.session_state[VIDEO_BYTES_KEY] = video_bytes
    elif path.is_file():
        st.session_state[VIDEO_BYTES_KEY] = path.read_bytes()
    return result


def analyse_upload(data: bytes, suffix: str, config: AppConfig) -> AnalysisResult:
    """Process an upload in an automatically removed temporary directory."""
    with tempfile.TemporaryDirectory(prefix="modelens-upload-") as directory:
        path = Path(directory) / f"capture{suffix}"
        path.write_bytes(data)
        return analyse_path(path, config, video_bytes=data)


def result_json_bytes(result: AnalysisResult) -> bytes:
    """Build a complete portable JSON export on demand."""
    payload: dict[str, Any] = analysis_result_to_dict(result)
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
