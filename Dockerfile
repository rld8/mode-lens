FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.11.33

RUN useradd --create-home --uid 10001 modelens
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev
COPY --chown=modelens:modelens . .
USER modelens
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1
CMD ["uv", "run", "--frozen", "--no-dev", "streamlit", "run", "src/modelens/interfaces/streamlit_app/Home.py", "--server.address=0.0.0.0"]

