# Changelog

All notable changes to openfusion are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Routing learning loop** — the heuristic router (`router.mode: heuristic`) now tracks an
  in-process EMA of fuse-vs-solo success rate per prompt tier (`openfusion.outcomes.OutcomeStore`)
  and nudges future decisions toward whichever has been winning once either side has enough
  observations. In-memory only (resets on restart); read-only snapshot at
  `GET /v1/routing/outcomes`.
- **Pipeline strategy** (`strategy: pipeline`) — chain sequential LLM steps, each feeding its
  output to the next via a `{step_name}` placeholder in the following step's system prompt. Each
  step is `use: solo` (single model) or `use: fuse` (full panel + judge synthesis); the last
  step's output streams to the caller. See `examples/pipeline.yaml.example`.
- **Classifier model routing** — with `router.mode: model` + `route_models`, one classifier call
  picks FUSE *or the specific model* for the prompt (a single decision instead of two), falling back
  to the difficulty heuristic on any error. Unified entry point `router.route_request`.
- **Cost preview** — `POST /v1/estimate` returns the number of calls, input-token estimate, output
  cap, and a `$` figure (from cached OpenRouter per-model pricing; best-effort, falls back to tokens).
  The playground shows `≈ N calls · ~$X` next to the Fuse button as you type.
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
- `/v1/chat/completions` now returns a clean 400 `invalid_request_error`
  ("Request body must be a JSON object") for a non-object JSON body (e.g. a
  list or string), instead of crashing with an unhandled 500 while recording
  the request outcome.
- `response_cache`'s cache key now folds in a fingerprint of the resolved panel/
  judge API key(s), so a `config_resolver`-based multi-tenant deployment (see
  `docs/EMBEDDING.md`) never serves one tenant's cached answer to another tenant
  sending an identical prompt against identically-named models.
- Router SOLO requests now forward the configured single model upstream instead
  of the literal string `"openfusion"`.
- A `pipeline`-strategy streaming request no longer crashes the SSE response on
  an upstream failure mid-pipeline. `pipeline_and_stream` now catches the
  failure and emits an SSE error chunk + `[DONE]`, matching how
  `vote_and_stream`/`ranked_and_stream`/`synthesize_and_stream` already
  degrade a panel or judge failure after the response has started.
- Per-key rate limiting (`limits.rate_limit_per_minute`) is no longer
  bypassable by rotating the `Authorization: Bearer` header on a deployment
  without a `gateway.api_keys` allowlist. The rate-limit identity is now only
  taken from a *validated* gateway token; unauthenticated traffic shares one
  `anonymous` bucket instead of each request minting its own budget.

### Changed
- The Docker image now runs `openfusion` as an unprivileged `openfusion` user
  instead of root.
- The playground's TypeScript client (`web/src/lib/api.ts`, `App.tsx`) now
  types request/response payloads (`ChatPayload`, `UsagePayload`) instead of
  `any`, catching shape mismatches at compile time.
- The playground's `App.tsx` test coverage went from 58% to 93% of statements
  (settings dialog, progress panel, panel grid, copy button, model-suggestion
  chips, and analysis/usage cards are now exercised); CI now runs `vitest`
  with coverage and fails under 90%/70%/85%/90% (statements/branches/
  functions/lines), matching the backend's existing coverage gate.

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
