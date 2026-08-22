# Scientific and operational limitations

- ModeLens is an educational laboratory, not a calibrated instrument, structural health
  monitor, certification system or safety diagnostic.
- A normal camera introduces compression, exposure changes, rolling shutter, timestamp
  uncertainty, quantisation and a finite sampling band. The synthetic renderer does not
  reproduce every one of these effects.
- The contour tracker assumes a bright, connected, slender specimen. Lucas–Kanade needs
  local texture and currently invalidates, rather than re-detects, lost stations.
- The v1 stabilisation signal is a quality indicator; it is not a full perspective or
  rolling-shutter correction.
- POD modes can mix when frequencies are close, excitation is weak or damping changes
  through the record.
- Log-decrement damping is returned only after minimum-peak and exponential-fit checks;
  `null` means not identifiable, not zero damping.
- Bootstrap frequency intervals reflect window-to-window variability. They exclude
  unmodelled calibration bias unless separately propagated.
- Euler–Bernoulli assumptions exclude thick/non-uniform beams, support flexibility,
  large deformation, torsion and added masses not represented by the uniform model.
- Only `EI/(rho*A)` is identifiable from frequencies without external geometry/density.
- Comparison labels describe this pair of experiments only. They cannot be translated to
  “safe”, “unsafe” or “damaged”.

