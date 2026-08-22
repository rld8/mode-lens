"""Convenience wrapper for the reproducible synthetic generator."""

from __future__ import annotations

import argparse
from pathlib import Path

from modelens.synthetic import SyntheticMode, SyntheticVideoSpec, generate_cantilever_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frequency", type=float, default=3.5)
    parser.add_argument("--damping", type=float, default=0.025)
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    spec = SyntheticVideoSpec(
        fps=args.fps,
        duration_s=args.duration,
        seed=args.seed,
        modes=(SyntheticMode(args.frequency, args.damping, 24.0),),
    )
    generate_cantilever_video(args.output, spec, args.output.with_suffix(".ground_truth.json"))


if __name__ == "__main__":
    main()
