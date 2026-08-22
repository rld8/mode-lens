from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from modelens.interfaces.cli import app
from modelens.synthetic import SyntheticMode, SyntheticVideoSpec, generate_cantilever_video


@pytest.mark.e2e
def test_analyze_command_exports_expected_artifacts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    video = tmp_path / "capture.mp4"
    output = tmp_path / "run"
    generate_cantilever_video(
        video,
        SyntheticVideoSpec(
            fps=60.0,
            duration_s=5.0,
            modes=(SyntheticMode(3.5, 0.025, 48.0),),
        ),
    )
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "--video",
            str(video),
            "--config",
            str(root / "configs/demo_cantilever.yaml"),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (output / "result.json").is_file()
    assert (output / "report.html").is_file()
    assert (output / "tracking_overlay.mp4").is_file()
