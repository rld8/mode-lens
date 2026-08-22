.PHONY: install demo quality test test-e2e benchmark docker clean

install:
	uv sync --all-groups
	uv run pre-commit install

demo:
	uv run streamlit run src/modelens/interfaces/streamlit_app/Home.py

quality:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src/modelens
	uv run bandit -q -r src/modelens
	mkdir -p .cache
	uv export --frozen --no-dev --no-hashes --format requirements-txt --output-file .cache/audit-requirements.txt
	uv run pip-audit --requirement .cache/audit-requirements.txt --cache-dir .cache/pip-audit

test:
	uv run pytest -m "not e2e"

test-e2e:
	uv run pytest -m e2e --no-cov

benchmark:
	uv run python scripts/benchmark_pipeline.py

docker:
	docker build -t modelens .

clean:
	rm -rf .cache .coverage .hypothesis .pytest_cache .mypy_cache .ruff_cache htmlcov build dist
	find src tests scripts -type d -name __pycache__ -prune -exec rm -rf {} +
