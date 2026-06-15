# AGENTS.md

## Cursor Cloud specific instructions

`openfusion` is a single, stateless Python (3.12) FastAPI service — an OpenAI-compatible
"compound-model" proxy. There is no database, cache, or queue. Dependencies are managed by
`pyproject.toml` and installed with `pip install -e ".[dev]"` (run automatically by the
startup update script).

### Running commands
- Console scripts (`openfusion`, `ruff`, `pytest`, `uvicorn`) install to `~/.local/bin`, which
  is **not on PATH** by default. Prefix commands with `export PATH="$HOME/.local/bin:$PATH"`,
  or invoke via `python3 -m` (e.g. `python3 -m pytest`, `python3 -m ruff check .`).
- Lint: `ruff check .` — Test: `pytest -v` (see `.github/workflows/ci.yml`). Tests need no
  network/secrets; upstream HTTP is mocked with `respx`.
- Run the server: `openfusion --host 0.0.0.0 --port 8000` (endpoints: `POST /v1/chat/completions`,
  `GET /v1/models`, `GET /healthz`).

### Config and end-to-end runs (non-obvious)
- The server loads `OPENFUSION_CONFIG` (default `openfusion.yaml`); if neither exists it falls
  back to `openfusion.yaml.example`. The example points at OpenRouter and requires
  `OPENROUTER_API_KEY` — config loading **raises** if a referenced `${ENV_VAR}` is unset.
- For a real end-to-end fusion request (`model: "openfusion"`) you need an OpenAI-compatible
  upstream. With no real key, point the config's `panel`/`judge`/`pass_through` at any local
  OpenAI-compatible server (e.g. a mock, Ollama, or vLLM) and set `OPENFUSION_CONFIG` to it.
  `GET /healthz` works without any upstream, but actual completions require one.
- Panel members are called **non-streaming**; the judge is called **streaming**. A mock upstream
  must support both modes of `POST /v1/chat/completions` for fusion to work.
- openfusion forwards the client request body to the upstream as-is. If the client omits
  `max_tokens`, the upstream applies its own (often large) default, which can fail on
  credit-limited provider keys (HTTP 402). For real OpenRouter runs on a low-credit key, pass a
  small `max_tokens` (e.g. `80`) in the request.
- tmux note: the tmux server may predate secret injection, so sessions started later won't see
  `OPENROUTER_API_KEY`. Start the session with `tmux ... new-session -e "OPENROUTER_API_KEY=$OPENROUTER_API_KEY"`
  (from a shell that already has the secret) before launching the server for real upstream runs.
