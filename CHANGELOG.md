# Changelog

All notable changes to openfusion are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`openfusion ask "…"`** — run a one-shot fusion from the terminal (streams the answer to stdout,
  progress to stderr); `--max-tokens` caps each call.
- **Live progress breakdown** — the playground (and the SSE `progress` events) now report each panel
  member as it finishes and the synthesis step, so you can see what you're waiting on.
- **Response-length control** — the playground Settings expose a max-tokens cap; per-request
  `openfusion.max_tokens` caps every panel/judge/pass-through call (clamped to 8192).

### Changed
- The judge prompt now asks for concise, focused answers, and the zero-config quick start caps
  responses at ~1024 tokens, so out-of-box answers aren't never-ending.

### Docs
- Document installing the CLI as a tool (`uv tool install .` / `pipx install .`) so `openfusion` is
  always on `PATH` without activating a venv, plus a Troubleshooting section and a `Makefile`.

### Added (continued)
- **Formatted answers** — the playground renders the response as Markdown (headings, lists, tables,
  code blocks via GitHub-flavored Markdown) with a copy button, instead of plain pre-wrapped text.
- **`openfusion setup`** — an interactive first-run wizard that prompts for your OpenRouter key and
  a recipe, then writes a private `openfusion.yaml`. The startup banner now nudges you to the
  playground (or `openfusion setup`) when no key is configured.

### Changed
- Moved the example configs from the repo root into `examples/` to declutter the top level.

### Fixed
- The playground now shows a clear "couldn't reach the server" message instead of a bare
  "Failed to fetch", and the API-key prompt reliably clears once a key is applied.

### Added (continued)
- **Zero-config quick start** — `openfusion` now boots with no config file and no env key (Budget
  preset), and `GET /` redirects to the playground. Paste your OpenRouter key in the UI to start.
- **Runtime API key** — `POST /v1/runtime/api-key` sets the upstream key in server memory (gated by
  `allow_ui_api_key`, on for the quick start). `GET /v1/config` reports `needs_api_key`.
- **React + Tailwind + shadcn playground** — the playground is now a proper SPA (source in `web/`,
  built assets shipped in the package), replacing the vanilla page.

### Changed
- The default-config fallback is the zero-config quick start instead of `examples/default.yaml.example`.

### Removed
- The static marketing landing page (`/`) and `docs/LANDING_PAGE.md`; `/` now redirects to the
  playground.

### Added (playground)
- **Interactive playground** at `GET /playground` — a zero-build single-page UI that talks only to
  the local `/v1` API (provider keys never reach the browser). Pick a Quality/Budget/Custom panel
  and judge, toggle web search, and watch the streamed answer, structured analysis, and token/cost.
- **`GET /v1/config`** — read-only active config (panel, judge, tools, presets, overrides allowed).
- **Per-request overrides** — optional `openfusion: { preset | panel | judge | tools }` request
  field (gated by `allow_request_overrides`, default off), mirroring OpenRouter Fusion's
  `analysis_models`/`model` plugin fields. Overrides reuse server credentials and stay bounded by
  auth, cost ceilings, and rate limits.
- **Production limits** (`limits`) — optional concurrency cap (`max_in_flight`, over-limit → 503)
  and per-gateway-key rate limiting (`rate_limit_per_minute`, over-limit → 429). Default unlimited.
- **Model-classifier routing** (`router.mode: model`) — the Auto Router can call a small classifier
  model to decide fuse-vs-solo, falling back to the heuristic on any error.
- **Ranked-choice aggregator** (`aggregator: ranked`) — one short judge call picks the single best
  panel answer; cheaper than synthesis, uses model judgment unlike majority vote.
- **Analysis transparency** (`analysis.emit`) — surface the judge's structured reasoning
  (consensus / contradictions / partial coverage / unique insights / blind spots) as a separate SSE
  `event: analysis` and an `analysis` field on non-streaming responses.
- **Prompt caching** (`cache.enabled`) — mark the self-fusion shared prefix with a `cache_control`
  breakpoint so repeated samples reuse a cached prompt where the provider supports it.
- **Auto Router** (`router.enabled`) — a per-prompt gate that answers simple prompts with a single
  pass-through call and reserves the panel for prompts that benefit (long, analytical, or code).
  Heuristic, no extra model call; `mode: heuristic | always | never`.
- **Debate strategy** (`strategy: debate`) — a diverse panel where each member revises after seeing
  the others' answers (`debate.rounds`) before the judge synthesizes.
- **Fusion-aware tool calling** — requests whose tools are server-executable
  (`openrouter:web_search`/`web_fetch`) now fuse instead of passing through; client-side function
  tools and mid-conversation tool turns still pass through.
- PyPI publish step in the release workflow, gated on a `PYPI_API_TOKEN` secret (no-op until set).
- Benchmark workflow bound to a protected `bench` environment for spend control.

### Added (earlier in this cycle)
- `preset: quality | budget` config switch — expands to a diverse OpenRouter
  panel + judge with web search/fetch enabled by default (the regime where
  synthesis beats the best single member per `bench/FINDINGS.md`). Mirrors
  OpenRouter Fusion's Quality/Budget UX. Explicit YAML always overrides a preset.
- `examples/preset.yaml.example` — one-line recipe to copy.
- "unique insights" added to the judge synthesis prompt, matching Fusion's
  synthesis dimensions (consensus / contradictions / partial coverage /
  unique insights / blind spots).
- Startup summary banner (recipe, panel, judge, tools, listen address, and the
  model name to call) printed by the CLI.
- Open-source project files: `CONTRIBUTING.md`, `SECURITY.md`,
  `CODE_OF_CONDUCT.md`, issue forms, PR template, Dependabot, `CITATION.cff`.
- README: CI/license/Python badges and an openfusion-vs-Fusion comparison table.

### Changed
- Friendlier configuration errors: a missing config file suggests copying an
  example, and a missing `${ENV_VAR}` suggests how to export it. The CLI prints
  a clean message and exits non-zero instead of dumping a traceback.

### Fixed
- `--config` is now honored by the uvicorn factory worker, not just the
  `OPENFUSION_CONFIG` environment variable.

## [0.1.0]

Initial MVP: OpenAI-compatible `POST /v1/chat/completions` + `/v1/models`,
parallel panel fan-out with graceful degrade, judge synthesis and majority-vote
aggregators, real SSE streaming with progress events, pass-through for
non-fusion models and tool calls, configurable cost ceilings, Prometheus
metrics at `/metrics`, agentic web tools for panel members, and a reproducible
`bench/` harness.

[Unreleased]: https://github.com/shahar-dagan/openfusion/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/shahar-dagan/openfusion/releases/tag/v0.1.0
