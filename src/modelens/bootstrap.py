"""Dependency composition for command-line and Streamlit interfaces."""

from __future__ import annotations

from modelens.adapters.jinja_report import JinjaReportRenderer
from modelens.adapters.opencv_tracking import BrightContourTracker, LucasKanadeTracker
from modelens.adapters.opencv_video import OpenCVVideoSource
from modelens.application.analyze_video import AnalyzeVideo
from modelens.config import AppConfig
from modelens.ports.reporting import ReportRenderer
from modelens.ports.tracking import FeatureTracker


def create_tracker(config: AppConfig) -> FeatureTracker:
    """Select the configured tracking implementation."""
    if config.tracking.method == "contour":
        return BrightContourTracker()
    return LucasKanadeTracker()


def create_analyzer(config: AppConfig) -> AnalyzeVideo:
    """Compose the complete local analysis use case."""
    return AnalyzeVideo(OpenCVVideoSource(), create_tracker(config))


def create_report_renderer() -> ReportRenderer:
    """Compose the self-contained HTML report adapter."""
    return JinjaReportRenderer()
