# Architecture

![ModeLens architecture](../assets/architecture.svg)

Dependencies point inward. `domain` contains validated entities and numerical services;
it imports neither OpenCV, Streamlit, the filesystem nor reporting code. `application`
orchestrates use cases through small ports. `adapters` implement those ports. `interfaces`
translate CLI or Streamlit interactions into application requests.

The deliberate extension seam is tracking: both Lucas–Kanade and a bright-contour
tracker satisfy the same contract. Stable equations remain functions instead of being
wrapped in empty interfaces. A run ID is derived from the input SHA-256 and validated
configuration, which connects exported arrays, JSON, HTML and logs.

Raw media is read, never rewritten. Persistence is an explicit export that stores only
derived arrays and the source hash/path; the Streamlit upload path is temporary.

