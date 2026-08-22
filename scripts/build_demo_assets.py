"""Build the two deterministic videos shipped with the repository."""

from __future__ import annotations

import json
from pathlib import Path

from modelens.synthetic import SyntheticMode, SyntheticVideoSpec, generate_cantilever_video


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    samples = root / "data/samples"
    baseline_spec = SyntheticVideoSpec()
    modified_spec = SyntheticVideoSpec(
        seed=43,
        modes=(
            SyntheticMode(3.10, 0.030, 48.0, 0.0),
            SyntheticMode(18.60, 0.020, 10.0, 0.35),
        ),
    )
    baseline = generate_cantilever_video(samples / "cantilever_baseline.mp4", baseline_spec)
    modified = generate_cantilever_video(samples / "cantilever_modified.mp4", modified_spec)
    payload = {
        "schema_version": "1.0",
        "description": "Synthetic ground truth for the bundled ModeLens demonstration.",
        "baseline": baseline,
        "modified": modified,
    }
    (samples / "expected_modes.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
