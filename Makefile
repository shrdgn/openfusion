.PHONY: install dev test lint ui run

# Install `openfusion` as a global tool (always on PATH, no venv to activate).
install:
	@command -v uv >/dev/null 2>&1 && uv tool install . || pipx install .

# Editable install for development (run inside an activated venv).
dev:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

# Rebuild the playground UI into openfusion/static/playground/.
ui:
	cd web && npm install && npm run build

# Start the server (zero-config; set OPENROUTER_API_KEY or use the playground).
run:
	openfusion --host 0.0.0.0 --port 8000
