# Changelog

All notable changes to openfusion are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Response cache** (`response_cache.enabled`) — identical fused requests (same prompt + recipe)
  are served instantly from an in-process TTL/LRU cache; cached responses are flagged `cached: true`
  and work for both streaming and non-streaming.
- **Usage callback** — `create_app(usage_callback=...)` fires per fused request with its usage/cost,
  for hosted metering (works for streaming via stream capture and non-streaming).

- **Embeddable engine** — `create_app(config_resolver=...)` resolves an
  `OpenFusionConfig` per request (the authenticated user's key + recipe), so the
  proxy can be wrapped for multi-tenant/hosted use without forking. A static
  config is now optional when a resolver is given. See `docs/EMBEDDING.md`.
- **Side-by-side panel view** — the playground shows each model's answer in its own card as the
  panel finishes, above the fused result. Backed by an opt-in `panel_answer` SSE event (per-request
  `openfusion.expose_panel`; off for the plain API so intermediate answers aren't exposed).
- **Model routing** — when `router.route_models` is set, the single-model branch
  picks the best model for the prompt by difficulty (cheap for easy, frontier for
  hard). `mode: never` gives pure routing; `mode: heuristic` fuses hard prompts
  and routes the rest. See `examples/route.yaml.example` and `ROADMAP.md`.

### Fixed
- Router SOLO requests now forward the configured single model upstream instead
  of the literal string `"openfusion"`.

## [0.1.0] — 2026-06-17

First public release. openfusion is an open, OpenAI-compatible mixture-of-agents
proxy — fan a prompt out to a panel of models, have a judge synthesize one
stronger answer — usable as a CLI chat, a web playground, or a drop-in API.

### Interfaces
- **Saved API key** — the CLI persists your OpenRouter key to
  `~/.config/openfusion/credentials` (chmod 600) on first entry, so `openfusion`
  doesn't re-prompt each run; `/key` re-enters it, and env vars take precedence.
- **`openfusion`** opens a Rich-rendered interactive chat REPL with the model
  panel: a banner, a spinner with live per-member panel progress, Markdown +
  syntax-highlighted answers, conversation history, and slash commands
  (`/preset`, `/tokens`, `/models`, `/clear`, `/help`).
- **`openfusion web`** (alias `serve`) starts the server + web playground, and opens the playground
  in your browser when run interactively (`--no-open` to disable; auto-skipped in Docker/CI/headless).
- **`openfusion ask "…"`** runs a one-shot fusion to stdout; `echo … | openfusion`
  pipes a one-shot too.
- **`openfusion setup`** is a guided first-run wizard.
- **Drop-in API** — `POST /v1/chat/completions` + `/v1/models`, real SSE
  streaming; `model: "openfusion"` triggers the panel, other models pass through.

### Playground (React + Tailwind + shadcn, shipped in the package)
- Quality / Budget / Custom panel, an editable model picker with autocomplete, a
  "Fuse with" judge, and a web-search toggle.
- Live progress stepper (per-model status → synthesis), Markdown-rendered answers
  with a copy button, the judge's structured analysis, and a token/cost readout.
- Paste your OpenRouter key in the UI (held only in server memory); a Settings
  dialog for the key, gateway token, and response-length cap.
- Zero-config quick start: boots the Budget preset, `GET /` redirects to the
  playground, `GET /v1/config` exposes the active recipe, `POST /v1/runtime/api-key`
  sets the key at runtime.

### Routing, strategies & aggregators
- **Presets** `quality | budget` — diverse OpenRouter panel + judge with web tools.
- **Auto Router** (`router.enabled`) — per-prompt fuse-vs-solo gate; heuristic or
  a small classifier model (`router.mode: model`).
- **Strategies** — `self_fusion`, `panel`, and `debate` (members revise after
  seeing each other's answers).
- **Aggregators** — `judge` synthesis (concise by default), `vote` (majority), and
  `ranked` (one judge call picks the best answer).
- **Analysis transparency** (`analysis.emit`) — emit the judge's structured
  reasoning as a separate SSE `analysis` event.
- **Fusion-aware tool calling** — server-executable web tools
  (`openrouter:web_search`/`web_fetch`) fuse; client function tools pass through.

### Operations & safety
- **Cost & length control** — `cost_controls` ceilings, per-request
  `openfusion.max_tokens`, and the playground response-length selector.
- **Production limits** (`limits`) — concurrency cap (→503) and per-key rate
  limiting (→429).
- **Prompt caching** (`cache.enabled`) for the self-fusion shared prefix.
- **Observability** — Prometheus metrics at `/metrics`; logs carry metadata only,
  never prompts or secrets.
- Graceful panel degradation, client-disconnect cancellation, and 429 backoff.

### Distribution & project
- `pip` / `uvx` / `uv tool` install, a Docker image, and a tag-driven release
  workflow (wheel + GHCR image; PyPI when `PYPI_API_TOKEN` is set).
- Open-source hygiene: `CONTRIBUTING`, `SECURITY`, `CODE_OF_CONDUCT`, issue/PR
  templates, Dependabot, `CITATION.cff`, and a reproducible `bench/` harness with
  honest findings in `bench/FINDINGS.md`.

[Unreleased]: https://github.com/shahar-dagan/openfusion/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/shahar-dagan/openfusion/releases/tag/v0.1.0
