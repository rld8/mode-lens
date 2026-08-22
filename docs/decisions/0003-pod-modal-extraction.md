# ADR 0003: POD/SVD before spectral peak selection

Status: accepted

A spectrum at one pixel discards spatial information. SVD separates energetic temporal
coordinates and spatial patterns without a trained model. Welch spectra and explicit
prominence thresholds then produce candidates. Close modes can still mix, so ModeLens
deduplicates resolution-equivalent peaks and reports uncertainty and quality flags.

