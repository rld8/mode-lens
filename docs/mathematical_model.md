# Mathematical model

## Signal matrix and POD

For `T` frames and `P` beam stations, ModeLens projects pixel motion onto the normal of
the initial beam axis and forms the centred matrix `X ∈ R^(T×P)`. Its singular value
decomposition is

\[
X = U\Sigma V^T.
\]

Columns of `UΣ` are temporal coordinates; rows of `V^T` are empirical spatial patterns.
Welch spectra of the coordinates provide candidate frequencies. POD is an energy
decomposition, so close or non-stationary modes can mix and must be marked ambiguous.
Candidates must pass both their component prominence and a configured global
`prominence × POD energy` detectability floor; this prevents tiny compression harmonics
from being reported as physical modes while keeping the threshold visible in every run.

## Modal Assurance Criterion

\[
MAC(\phi_a,\phi_b)=
\frac{|\phi_a^T\phi_b|^2}{(\phi_a^T\phi_a)(\phi_b^T\phi_b)}.
\]

The implementation rejects zero vectors and clips floating-point round-off to `[0,1]`.
MAC measures collinearity only.

## Damping

A narrow-band modal coordinate is expected to have envelope
`A(t)=A0 exp(-ζω_n t)`. ModeLens regresses the logarithm of at least five positive peaks
and rejects positive slopes or poor exponential fit. The decay rate `r` gives

\[
\zeta=\frac{r}{\sqrt{\omega_d^2+r^2}}.
\]

## Euler–Bernoulli cantilever

For a uniform, slender rectangular beam under small transverse deflection,

\[
f_n=\frac{\beta_n^2}{2\pi L^2}\sqrt{\frac{EI}{\rho A}},\qquad
I=\frac{bh^3}{12},\quad A=bh.
\]

The cantilever roots begin `1.875104`, `4.694091`, `7.854757`. Frequencies alone identify
the combined parameter `EI/(ρA)`. Young's modulus is derived only when geometry and
density are fixed by external measurements. Shear deformation, rotary inertia,
non-uniform mass, large deflection and support compliance are outside the v1 model.

## Sampling and uncertainty

The nominal Nyquist limit is `FPS/2`; ModeLens uses a conservative observable maximum
of `0.4×FPS`. Bootstrap intervals resample spectral windows and can sample configured
FPS and pixel-scale standard uncertainties with a recorded seed. Pixel scale cancels
from frequency, damping ratios and normalized shapes, but remains relevant to exported
displacement amplitudes. The twin interval additionally samples modal-frequency and
free-length uncertainty. These intervals still do not represent unmodelled camera bias.
