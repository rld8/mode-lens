"""Benchmark the fixed demo without claiming hardware-independent performance."""

from __future__ import annotations

import json
import platform
import resource
import sys
import time
import tracemalloc
from pathlib import Path

from modelens.application.analyze_video import AnalyzeVideoRequest
from modelens.bootstrap import create_analyzer
from modelens.config import load_config
from modelens.synthetic import SyntheticVideoSpec, generate_cantilever_video


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    video = root / "data/samples/cantilever_baseline.mp4"
    if not video.exists():
        generate_cantilever_video(video, SyntheticVideoSpec())
    config = load_config(root / "configs/demo_cantilever.yaml")
    tracemalloc.start()
    started = time.perf_counter()
    result = create_analyzer(config).execute(AnalyzeVideoRequest(video, config))
    elapsed = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    process_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    process_peak_rss_bytes = (
        process_peak_rss if sys.platform == "darwin" else process_peak_rss * 1024
    )
    expected = [mode.frequency_hz for mode in SyntheticVideoSpec().modes]
    measured = [mode.frequency_hz for mode in result.modal.modes[: len(expected)]]
    errors = [abs(value - truth) / truth for value, truth in zip(measured, expected, strict=True)]
    payload = {
        "run_id": result.run_id,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "video": str(video.relative_to(root)),
        "elapsed_s": elapsed,
        "python_peak_memory_mb": peak_bytes / (1024.0 * 1024.0),
        "process_peak_rss_mb": process_peak_rss_bytes / (1024.0 * 1024.0),
        "expected_frequency_hz": expected,
        "measured_frequency_hz": measured,
        "relative_frequency_error": errors,
        "note": (
            "Peak RSS includes the complete benchmark process; tracemalloc reports "
            "Python allocations."
        ),
    }
    destination = root / "runs/benchmark.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
