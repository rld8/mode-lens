# Requirements traceability

This matrix maps the v1 functional requirements to code and executed evidence. It is a
traceability aid, not a substitute for the scientific limitations.

| Requirement | Implementation | Verification |
|---|---|---|
| ML-F01 video input | `OpenCVVideoSource.metadata/frames`, CLI and Streamlit upload | invalid-input unit branches; OpenCV integration; CLI E2E |
| ML-F02 experiment configuration | immutable Pydantic models, YAML, UI axis/ROI/scale/geometry/material controls | config validation tests; demo configuration parsed in every integration run |
| ML-F03 capture quality | decoded/effective FPS, duration, resolution, blur, contrast, saturation and background motion | OpenCV integration plus report/UI quality tables |
| ML-F04 at least 12 points | both trackers enforce survivor count; validity/confidence per observation | contour pipeline and stabilised Lucas–Kanade integration tests |
| ML-F05 signal preprocessing | bounded gaps, Hampel, detrend, zero-phase Butterworth; raw retained | positive/negative/boundary unit tests |
| ML-F06 modal identification | POD/SVD, Welch peaks with interpolated frequency, damping checks and seeded intervals | analytic synthetic arrays and 720p video benchmark |
| ML-F07 twin calibration | weighted `EI/(rho*A)` fit and Monte Carlo interval; optional derived `E` with fixed inputs | analytic parameter round-trip tests |
| ML-F08 comparison | MAC/frequency assignment and conservative labels | unit tests plus saved synthetic baseline/modified comparison |
| ML-F09 interaction | five-page Streamlit app, selectable modes, animation, twin controls, explicit actions and error/empty states | HTTP health smoke check; algorithms exercised below the UI |
| ML-F10 export | complete JSON, NPZ, signals/trajectories CSV, overlay MP4 and self-contained HTML | repository/report integration and CLI E2E |

## Non-functional evidence

- The fixed 10 s, 1280×720, 120 FPS demo analysed in 25.964 s on the recorded host,
  below the 45 s target.
- Peak process RSS was 281.875 MiB, below the 2 GiB budget. Trackers consume decoded
  frames as a stream rather than retaining the full video in RAM.
- Uploads use an automatically removed temporary directory; persistence occurs only via
  an explicit export action.
- Numerical entities reject inconsistent shapes and non-finite cleaned results. Missing
  damping/intervals remain `null` plus a reason.
- Run IDs bind the input SHA-256 to the full validated configuration and recorded seed.
- Native Streamlit widgets support keyboard navigation and the custom report uses
  semantic HTML and high-contrast colours. A formal independent WCAG audit was not
  performed, so AA conformance is not claimed.
- Docker build verification remains pending on a host with Docker; see the validation
  report. CI is configured to build the image.

