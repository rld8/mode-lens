"""Self-contained HTML reporting adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import plotly.io as pio
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from modelens.adapters.plotly_presenter import (
    mac_figure,
    mode_shape_figure,
    psd_figure,
    scree_figure,
    signal_figure,
    twin_figure,
)
from modelens.application.dto import analysis_result_to_dict
from modelens.domain.entities import AnalysisResult, ComparisonResult


class JinjaReportRenderer:
    """Render portable HTML with Plotly bundled into the document."""

    def __init__(self, template_directory: Path | None = None) -> None:
        """Load the report template from the repository or a supplied directory."""
        packaged = Path(__file__).resolve().parents[1] / "templates"
        checkout = Path(__file__).resolve().parents[3] / "templates"
        root = template_directory or (packaged if packaged.is_dir() else checkout)
        self._environment = Environment(
            loader=FileSystemLoader(root),
            autoescape=select_autoescape(["html", "xml"]),
            undefined=StrictUndefined,
        )

    def render_analysis(self, result: AnalysisResult, destination: Path) -> Path:
        """Render an analysis report and embed all data required by its figures."""
        template = self._environment.get_template("report.html.j2")
        figures = [
            self._html(signal_figure(result), include_plotly=True),
            self._html(psd_figure(result)),
            self._html(scree_figure(result)),
        ]
        figures.extend(self._html(mode_shape_figure(result, mode)) for mode in result.modal.modes)
        if result.twin is not None:
            figures.append(self._html(twin_figure(result)))
        context: dict[str, Any] = {
            "title": f"ModeLens analysis — {result.run_id}",
            "subtitle": "Video-to-modal digital twin",
            "run_id": result.run_id,
            "warning": (
                "Educational experiment only; not a structural certification or safety diagnosis."
            ),
            "modes": result.modal.modes,
            "quality": result.quality,
            "twin": result.twin,
            "figures": figures,
            "comparison": None,
            "result_json": json.dumps(
                analysis_result_to_dict(result, include_series=False), indent=2, sort_keys=True
            ),
        }
        return self._write(template.render(**context), destination)

    def render_comparison(self, result: ComparisonResult, destination: Path) -> Path:
        """Render a comparison report with a MAC heatmap and permitted labels."""
        template = self._environment.get_template("report.html.j2")
        payload = {
            "matches": [
                {
                    "baseline_index": match.baseline_index,
                    "modified_index": match.modified_index,
                    "mac": match.mac,
                    "relative_frequency_change": match.relative_frequency_change,
                    "damping_change": match.damping_change,
                    "label": match.label.value,
                    "reason": match.reason,
                }
                for match in result.matches
            ],
            "mac_matrix": result.mac_matrix.tolist(),
            "unmatched_baseline": list(result.unmatched_baseline),
            "unmatched_modified": list(result.unmatched_modified),
            "warnings": list(result.warnings),
        }
        return self._write(
            template.render(
                title="ModeLens controlled comparison",
                subtitle="Baseline vs modified experiment",
                run_id="comparison",
                warning="Measured change in this test; not a structural diagnosis.",
                modes=(),
                quality=None,
                twin=None,
                figures=[self._html(mac_figure(result), include_plotly=True)],
                comparison=result,
                result_json=json.dumps(payload, indent=2, sort_keys=True),
            ),
            destination,
        )

    @staticmethod
    def _html(figure: Any, include_plotly: bool = False) -> str:
        return str(
            pio.to_html(
                figure,
                full_html=False,
                include_plotlyjs=bool(include_plotly),
                config={"displaylogo": False, "responsive": True},
            )
        )

    @staticmethod
    def _write(html: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(html, encoding="utf-8")
        temporary.replace(destination)
        return destination
