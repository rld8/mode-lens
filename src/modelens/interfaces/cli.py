"""Reproducible ModeLens command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from modelens.adapters.local_repository import LocalExperimentRepository
from modelens.adapters.opencv_tracking import render_tracking_overlay
from modelens.application.analyze_video import AnalyzeVideoRequest
from modelens.application.compare_experiments import CompareExperiments
from modelens.application.export_report import ExportReport
from modelens.bootstrap import create_analyzer, create_report_renderer
from modelens.config import load_config
from modelens.domain.errors import ModeLensError
from modelens.synthetic import SyntheticMode, SyntheticVideoSpec, generate_cantilever_video

app = typer.Typer(no_args_is_help=True, help="Explainable modal analysis from ordinary video.")
console = Console()


def _analyse(video: Path, config_path: Path, output: Path) -> Path:
    config = load_config(config_path)
    analyzer = create_analyzer(config)
    result = analyzer.execute(AnalyzeVideoRequest(video_path=video, config=config))
    repository = LocalExperimentRepository()
    result_file = repository.save(result, output)
    render_tracking_overlay(str(video), str(output / "tracking_overlay.mp4"), result.tracking)
    ExportReport(create_report_renderer()).analysis(result, output / "report.html")
    table = Table(title=f"ModeLens run {result.run_id}")
    table.add_column("Mode")
    table.add_column("Frequency [Hz]")
    table.add_column("Damping")
    table.add_column("95% frequency interval")
    for mode in result.modal.modes:
        table.add_row(
            str(mode.index),
            f"{mode.frequency_hz:.4f}",
            "not identifiable" if mode.damping_ratio is None else f"{mode.damping_ratio:.4f}",
            "n/a"
            if mode.frequency_ci_hz is None
            else f"{mode.frequency_ci_hz[0]:.3f}–{mode.frequency_ci_hz[1]:.3f}",
        )
    console.print(table)
    console.print(f"Artifacts: {output.resolve()}")
    return result_file


@app.command()
def analyze(
    video: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/demo_cantilever.yaml"),
    output: Annotated[Path, typer.Option(file_okay=False)] = Path("runs/analysis"),
) -> None:
    """Analyse a configured vibration video and export auditable artifacts."""
    try:
        _analyse(video, config, output)
    except (ModeLensError, ValueError, OSError) as exc:
        console.print(f"[red]Analysis failed:[/red] {exc}")
        raise typer.Exit(code=2) from exc


@app.command()
def compare(
    baseline: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    modified: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(file_okay=False)] = Path("runs/comparison"),
) -> None:
    """Compare two complete ModeLens result files."""
    repository = LocalExperimentRepository()
    try:
        result = CompareExperiments().execute(repository.load(baseline), repository.load(modified))
    except (ValueError, OSError) as exc:
        console.print(f"[red]Comparison failed:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
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
    (output / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ExportReport(create_report_renderer()).comparison(result, output / "report.html")
    console.print(f"Comparison artifacts: {output.resolve()}")


@app.command()
def synthetic(
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path("data/raw/synthetic.mp4"),
    frequency: Annotated[float, typer.Option(min=0.1)] = 3.5,
    damping: Annotated[float, typer.Option(min=0.0, max=0.3)] = 0.025,
    fps: Annotated[float, typer.Option(min=15.0)] = 120.0,
    duration: Annotated[float, typer.Option(min=2.0, max=30.0)] = 8.0,
    seed: int = 42,
) -> None:
    """Generate a labelled single-mode cantilever video with known truth."""
    spec = SyntheticVideoSpec(
        fps=fps,
        duration_s=duration,
        seed=seed,
        modes=(SyntheticMode(frequency, damping, 48.0),),
    )
    truth_path = output.with_suffix(".ground_truth.json")
    generate_cantilever_video(output, spec, truth_path)
    console.print(f"Synthetic video: {output.resolve()}")
    console.print(f"Ground truth: {truth_path.resolve()}")


@app.command()
def demo(
    output: Annotated[Path, typer.Option(file_okay=False)] = Path("runs/demo"),
) -> None:
    """Run the included baseline video through the complete pipeline."""
    video = Path("data/samples/cantilever_baseline.mp4")
    if not video.exists():
        generate_cantilever_video(
            video, SyntheticVideoSpec(), Path("data/samples/expected_modes.json")
        )
    try:
        _analyse(video, Path("configs/demo_cantilever.yaml"), output)
    except (ModeLensError, ValueError, OSError) as exc:
        console.print(f"[red]Demo failed:[/red] {exc}")
        raise typer.Exit(code=2) from exc


if __name__ == "__main__":
    app()
