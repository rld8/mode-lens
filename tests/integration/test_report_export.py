from __future__ import annotations

from modelens.adapters.jinja_report import JinjaReportRenderer
from modelens.adapters.local_repository import LocalExperimentRepository
from modelens.application.compare_experiments import CompareExperiments
from modelens.application.export_report import ExportReport
from modelens.domain.entities import AnalysisResult


def test_result_round_trip_and_self_contained_report(
    tmp_path, analysis_result: AnalysisResult
) -> None:
    repository = LocalExperimentRepository()
    result_path = repository.save(analysis_result, tmp_path / "run")
    loaded = repository.load(result_path)
    assert loaded.run_id == analysis_result.run_id
    assert (
        loaded.signal.cleaned_displacement.shape
        == analysis_result.signal.cleaned_displacement.shape
    )
    exporter = ExportReport(JinjaReportRenderer())
    report = exporter.analysis(loaded, tmp_path / "run/report.html")
    html = report.read_text(encoding="utf-8")
    assert "plotly.js" in html
    assert "not a structural certification" in html
    assert (tmp_path / "run/signals.csv").is_file()
    assert (tmp_path / "run/trajectories.csv").is_file()
    comparison = CompareExperiments().execute(loaded, loaded)
    comparison_report = exporter.comparison(comparison, tmp_path / "comparison.html")
    assert "Modal Assurance Criterion" in comparison_report.read_text(encoding="utf-8")
