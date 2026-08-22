# ADR 0002: Sparse pyramidal Lucas–Kanade plus a contour fallback

Status: accepted

Sparse Lucas–Kanade is explainable, CPU-friendly and appropriate for local texture. A
forward-backward pass invalidates inconsistent observations. Synthetic and high-contrast
captures use a second adapter that estimates a bright component centreline. The fallback
is not silently selected: the run records the configured method.

