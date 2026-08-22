# ModeLens — Video-to-Modal Digital Twin

> Turn an ordinary vibration video into an explainable modal experiment.

[![Python 3.12](https://img.shields.io/badge/Python-3.12-35618f)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-35618f)](LICENSE)

ModeLens is a local Python laboratory for analysing the vibration of a simple flexible
specimen, such as a cantilever ruler. It tracks multiple points from video, cleans their
displacement signals, separates spatial patterns with POD/SVD, estimates modal frequency
and damping, and calibrates the identifiable parameter of an Euler–Bernoulli beam twin.

The complete analysis remains auditable: each result is linked to the source-video hash,
tracking confidence, preprocessing record, spectrum, spatial shape, uncertainty interval
and physical-model residual.

> **Scope:** educational and controlled experiments only. ModeLens is not calibrated
> instrumentation, structural certification or a safety or damage diagnostic.

![Clean architecture overview](assets/architecture.svg)

## Verified repository snapshot

The bundled videos are generated locally from recorded ground truth; they are not
third-party or real-world measurements. The validation report included in this repository
records the following results:

| Check | Measured result |
| --- | ---: |
| Unit/integration tests | 37 passed; separate E2E smoke test passed |
| Coverage | 86.04% of the configured testable source |
| Baseline frequencies | 3.499963 Hz and 21.910112 Hz |
| Baseline relative frequency error | 0.001043% and 0.046175% |
| 720p/10 s demo analysis runtime | 25.964 s |
| Peak process RSS | 281.875 MiB |

These measurements apply to the synthetic assets and recorded environment only. No
real-specimen repeatability is claimed. Exact versions, hashes, commands and comparison
results are available in [`docs/validation_report.md`](docs/validation_report.md), with
machine-readable benchmark data under `docs/validation/`.

## Local installation

Requirements: Python 3.12, [`uv`](https://docs.astral.sh/uv/) and a CPU. No API key, GPU,
cloud account or external dataset is required.

```bash
git clone https://github.com/rld8/mode-lens.git
cd mode-lens
uv sync --all-groups
uv run python scripts/build_demo_assets.py
uv run modelens demo --output runs/demo
```

Open `runs/demo/report.html`, then launch the interactive application:

```bash
make demo
```

Streamlit will expose the application at <http://127.0.0.1:8501/>. The five pages cover
capture quality, tracking, modal analysis, the restricted digital twin and
baseline/modified comparison.

## Docker deployment

The repository contains a `Dockerfile` for running the Streamlit interface on port 8501:

```bash
make docker
docker run --rm -p 8501:8501 modelens
```

Open <http://127.0.0.1:8501/> and verify the health endpoint at
<http://127.0.0.1:8501/_stcore/health>.

The current validation report does not record a completed Docker build, so the container
must be built and checked on the target host before presenting that deployment as
validated. The supported verified path in the recorded snapshot is the local `uv`
installation.

## Reproducible commands

```bash
uv run modelens analyze \
  --video data/samples/cantilever_baseline.mp4 \
  --config configs/demo_cantilever.yaml \
  --output runs/baseline

uv run modelens analyze \
  --video data/samples/cantilever_modified.mp4 \
  --config configs/demo_cantilever.yaml \
  --output runs/modified

uv run modelens compare \
  --baseline runs/baseline/result.json \
  --modified runs/modified/result.json \
  --output runs/comparison

uv run modelens synthetic \
  --frequency 3.5 \
  --damping 0.025 \
  --output data/raw/synthetic.mp4

make quality
make test
make test-e2e
make benchmark
```

Each analysis exports:

- `result.json`: complete versioned domain result;
- `arrays.npz`: compact numerical arrays;
- `signals.csv` and `trajectories.csv`: interoperable derived data;
- `tracking_overlay.mp4`: visual tracking audit;
- `report.html`: self-contained report with Plotly embedded.

Raw user media is read in place and is never copied into the run directory.

## Scientific pipeline

1. Decode the media metadata and reject corrupt, oversized or overlong input.
2. Measure blur, contrast, saturation and apparent global camera movement.
3. Track stations with pyramidal Lucas–Kanade and a forward/backward check, or use the
   explicitly selected bright-contour adapter for a high-contrast specimen.
4. Project movement onto the normal axis while retaining the raw signal and validity
   mask.
5. Interpolate bounded internal gaps, apply Hampel outlier rejection, detrend linearly
   and use a zero-phase Butterworth band-pass filter.
6. Decompose the time × station matrix with SVD/POD and inspect Welch spectra of the modal
   coordinates.
7. Estimate damping only when at least five peaks follow an approximately exponential
   decay; otherwise return `null` with a reason.
8. Bootstrap spectral windows with a recorded seed to estimate frequency intervals.
9. Fit `EI/(rho*A)` and report residuals. Derive `E` only when geometry and density are
   supplied externally.
10. Pair two experiments using frequency and Modal Assurance Criterion (MAC), returning
    only `stable`, `measurable_change` or `inconclusive`.

Equations, units and assumptions are documented in
[`docs/mathematical_model.md`](docs/mathematical_model.md). The mapping between functional
requirements and evidence is recorded in
[`docs/requirements_traceability.md`](docs/requirements_traceability.md).

## Architecture

```text
interfaces (Streamlit, Typer)
        │
        ▼
application use cases ─────► ports
        │                      ▲
        ▼                      │
domain services          adapters (OpenCV, files, Jinja/Plotly)
```

Dependencies point inward. The domain has no OpenCV, Streamlit, filesystem or reporting
dependency. Tracking is the main extension boundary: changing the tracker does not modify
the `AnalyzeVideo` use case. Stable mathematical functions remain direct functions rather
than being hidden behind empty interfaces. Design decisions are documented under
`docs/decisions/`.

## Recording an experiment

Use a plastic ruler or a thin wooden strip, a high-contrast background, a stable phone at
60 or 120 fps and a scale in the same image plane. Keep 15–20% of the specimen clamped,
record 5–10 seconds, deflect gently and release. Capture three repetitions before and
after any controlled modification.

The complete capture, safety and acceptance procedure is in
[`docs/experiment_protocol.md`](docs/experiment_protocol.md). Camera compression, rolling
shutter, quantisation, timestamp error and perspective can bias the result. If a mode
approaches `0.4 × FPS`, tracking is unstable or uncertainty cannot be estimated, the
result must be treated as inconclusive. See
[`docs/limitations.md`](docs/limitations.md).

## Validation strategy

- Analytic checks for cantilever frequencies, shapes, MAC and parameter recovery.
- Property tests for MAC scale/sign invariance and numerical bounds.
- Synthetic arrays with two known modes, damping, noise and a fixed seed.
- Generated video passed through OpenCV tracking and modal extraction.
- JSON/CSV/NPZ/report round-trip integration.
- Typer command smoke test and a fixed demo benchmark.
- Ruff formatting and linting, strict mypy, Bandit and GitHub Actions.

The synthetic generator records its ground truth and rejects frequencies at or above
Nyquist. MP4 hashes can vary between codecs, so numerical acceptance is based on decoded
motion and documented tolerances rather than on identical compressed bytes.

The recorded synthetic baseline/modified comparison paired both modes. Mode 1 was labelled
`measurable_change` (`Δf/f = -11.426%`, `MAC = 0.999975`). Mode 2 was labelled
`inconclusive`: its point estimate changed by `-15.121%`, but its deliberately broad
intervals overlapped. These are synthetic pipeline checks, not physical findings.

## Project structure

```text
mode-lens/
├── configs/                 # validated experiment and processing settings
├── data/samples/            # reproducibly generated demo and ground truth
├── docs/                    # guide, protocol, mathematics, ADRs and limitations
├── scripts/                 # sample generation and measured benchmark
├── src/modelens/
│   ├── domain/              # entities, signals, modes, beam and comparison
│   ├── application/         # use cases and boundary DTOs
│   ├── ports/               # video, tracking, repository and report contracts
│   ├── adapters/            # OpenCV, local artifacts, Plotly and Jinja
│   └── interfaces/          # Typer CLI and five-page Streamlit application
├── templates/               # self-contained report template
└── tests/                   # unit, property, integration and E2E checks
```

## References

- [OpenCV optical-flow tutorial](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)
- [SciPy signal-processing API](https://docs.scipy.org/doc/scipy/reference/signal.html)
- [Streamlit multipage navigation](https://docs.streamlit.io/develop/concepts/multipage-apps/page-and-navigation)
- [uv dependency groups](https://docs.astral.sh/uv/concepts/projects/dependencies/)

These references document the software APIs. The physical assumptions used by ModeLens
are stated in the mathematical model and checked against analytic synthetic data.

## Roadmap

- Re-detection of lost Lucas–Kanade stations and optional affine stabilisation output.
- Half-power damping as a second estimator with agreement diagnostics.
- Monte Carlo propagation of measured scale/FPS uncertainty into the twin.
- A small, consented real-video dataset with three-repeat reports.

## Contributing, citation and licence

Read [`CONTRIBUTING.md`](CONTRIBUTING.md). Citation metadata is provided in
[`CITATION.cff`](CITATION.cff). The code is available under the [MIT License](LICENSE).
User and third-party videos retain their own rights and must not be committed without
permission.
