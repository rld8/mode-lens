# ADR 0001: Clean boundaries around scientific I/O

Status: accepted

The domain must be executable with NumPy/SciPy arrays alone. Video decoding, tracking,
persistence and reporting are ports because their implementations or operating context
can change. Stable numerical formulae remain functions. This keeps tests small and makes
an alternative tracker possible without editing the analysis use case.

