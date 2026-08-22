# Contributing

Use Python 3.12 and `uv sync --all-groups`. Create a focused branch, add positive,
negative and boundary tests, and run `make quality`, `make test` and the relevant smoke
test. Scientific changes must state units, assumptions, a reference or analytic ground
truth, and the tolerance used. Never commit private videos, raw user data or invented
benchmark values.

