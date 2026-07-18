# openfusion Architecture

This document explains module boundaries, extension points, and security decisions for openfusion.

## Why these modules exist

openfusion is a thin FastAPI proxy. Each module owns one concern so strategies, routers, and eval harnesses can grow without rewrites.

| Module | Responsibility | Must NOT do |
|--------|----------------|-------------|
| `server.py` | HTTP routes, auth gate, routing (`openfusion` vs pass-through), cancellation orchestration | SSE framing, judge prompt logic |
| `config.py` | Typed config from YAML + env | HTTP or upstream calls |
| `overrides.py` | Per-request panel/judge/preset/tools overrides; fills in the runtime UI API key | HTTP, SSE framing |
| `cost.py` | Token ceilings and request cost policy | Provider-specific pricing math |
| `pricing.py` | Cached best-effort per-model `$` pricing from the upstream `/models` endpoint | Business logic about panels or judges |
| `estimate.py` | Pre-run cost/usage estimate for `POST /v1/estimate` (calls, tokens, `$`) | HTTP, upstream calls |
| `router.py` | Per-prompt fuse-vs-solo decision (heuristic or model classifier) | SSE framing |
| `outcomes.py` | In-process EMA of fuse/solo success rate per prompt tier; nudges `router.py`'s heuristic | HTTP, persistence |
| `limits.py` | Concurrency cap + per-key rate limiting | HTTP, prompt/secret handling |
| `responsecache.py` | In-process TTL/LRU cache of fused answers, keyed by prompt + recipe | HTTP, upstream calls |
| `cache.py` | Prompt-cache breakpoint marking for the shared prefix | HTTP or upstream calls |
| `upstream.py` | Shared httpx client for OpenAI-compatible APIs | Business logic about panels or judges |
| `panel.py` | Parallel fan-out, timeouts, degrade, 429 retry, debate rounds | SSE framing |
| `pipeline.py` | Sequential `strategy: pipeline` steps (`solo`/`fuse`), injecting each step's output into the next via `{step_name}` | SSE framing |
| `synthesize.py` / `vote.py` / `ranked.py` | Aggregators: judge prompt assembly + text deltas, majority vote, judge pick | SSE framing |
| `stream.py` | All OpenAI chunk/SSE framing, progress events, terminal usage | Judge prompt content decisions |
| `metrics.py` | In-process counters/latency/token+cost registry, Prometheus text rendering | HTTP, upstream calls, prompt/secret handling |
| `tools.py` | Injects OpenRouter server-side web-search/web-fetch tools into request bodies | Client-side function-tool execution |
| `errors.py` | OpenAI-compatible error types and response helpers | HTTP routing, upstream calls |
| `credentials.py` | Local CLI credential storage (`~/.config/openfusion/credentials`, `600` perms) | Server-side key handling |

## Request flow

```
Client → server.py
           ├─ model != openfusion → upstream.py (pass-through)
           ├─ client function tools / tool-call turn → upstream.py (pass-through, no fusion)
           ├─ router.route() == SOLO → upstream.py (pass-through, single model)
           └─ model == openfusion (incl. server-executable web tools)
                 ├─ strategy == pipeline → pipeline.run_pipeline()
                 │                          (chains solo/fuse steps, streams the last step)
                 └─ else → panel.gather_panel()   (debate strategy: + revision rounds)
                             → stream.synthesize_and_stream()
                                   → synthesize.synthesize() (deltas)
                                   → stream wraps deltas into SSE
```

Tool handling: `server._requires_pass_through_tools` distinguishes tools the upstream executes
server-side (`openrouter:web_search`/`web_fetch`, which fuse) from client-side function tools and
mid-conversation tool turns (which pass through, since their results return through the client).

## Extension points

1. **Synthesis strategies** — `strategy` selects how the panel is produced (`self_fusion`, `panel`, `debate`, `pipeline`); `aggregator` selects how answers combine (`judge`, `vote`). Add a new strategy by extending `panel.expand_panel_members` / `gather_panel`, or a new aggregator alongside `synthesize`/`vote`. `pipeline` is a parallel code path (`pipeline.run_pipeline`) rather than a panel/aggregator variant — it chains `solo`/`fuse` steps sequentially instead of fusing one panel.
2. **Router gate** — `router.route()` runs before `gather_panel` in `server.py`. Today it is a heuristic; swap in an LLM classifier behind the same `RouteDecision` return type.
3. **Eval harness** — `bench/` calls the same HTTP surface as production clients; no special internal APIs.
4. **Embedding** — `create_app(config_resolver=...)` resolves config per request for multi-tenant/hosted wrappers; see [docs/EMBEDDING.md](EMBEDDING.md).

## Web UI

The playground is a React + Tailwind + shadcn SPA. Source lives in `web/`; `vite build` writes
hashed assets into `openfusion/static/playground/`, which are committed and shipped in the wheel so
`pip`/`uvx` users get the UI with no Node toolchain. The server mounts it at `/playground` (and `/`
redirects there). It only calls the local `/v1` API — never provider APIs — so provider keys stay
server-side. `GET /v1/config` exposes the active panel/judge and onboarding flags; `POST
/v1/runtime/api-key` sets the upstream key in memory when `allow_ui_api_key` is on.

## Configuration and secrets

- Runtime config lives in `openfusion.yaml` (gitignored). Use `examples/default.yaml.example` as the template.
- `examples/dev.yaml.example` is the low-cost live-test recipe; it is intentionally smaller than
  the default self-fusion example.
- `${ENV_VAR}` placeholders in YAML are expanded at load time; missing env vars fail fast.
- Upstream provider API keys come from config/env only. Client `Authorization` is an optional openfusion gateway token.
- `cost_controls` sets pass-through, panel, and judge token ceilings. Visible over-limit requests
  fail with `400`; internal panel calls clamp because panel output is intermediate.

## Observability

- `upstream.py` emits one structured log line per upstream request with phase, label, model, stream
  mode, status, latency, chunk count, and provider usage/cost when returned.
- `metrics.py` aggregates those same events into cumulative series exposed at `GET /metrics` in
  Prometheus text format. Recording happens at two chokepoints — `upstream._log_request` (per
  upstream call) and the `server.py` route handler (per client-facing request, with accurate
  end-to-end latency for streaming via the generator's `finally`). Panel success/failure counts are
  recorded in `panel.gather_panel`.
- Metrics carry only labels (`route`, `phase`, `kind`, `outcome`) and numbers — never prompts,
  labels derived from user content, or secrets. `/metrics` is unauthenticated; treat it as
  scrape-only and bind it to a trusted interface.
- Logs must not include prompts, response text, `Authorization`, or `api_key` values.
- Usage and cost numbers are provider-reported and best-effort; missing provider usage is omitted.

## Security considerations

| Concern | Mitigation | Follow-up |
|---------|------------|-----------|
| Upstream key exfiltration | Never read provider keys from client headers or body | Audit logs for accidental key emission |
| Gateway auth bypass | Optional `OPENFUSION_API_KEYS` / `gateway.api_keys` allowlist | Rate limiting per gateway key |
| Secret logging | Redact `Authorization` and `api_key` fields in debug logs | Structured log scrubber |
| Prompt leakage in logs | Upstream request logs include metadata/usage only, not request or response bodies | Add automated log schema checks for every route |
| Accidental credit burn | `cost_controls` inject/reject/clamp `max_tokens`; live smoke requires explicit opt-in | Per-key budget counters and rate limits |
| Config file permissions | Document `chmod 600 openfusion.yaml` in README | Optional startup warning if world-readable |
| SSRF via `base_url` | Config is operator-controlled; document trust boundary | Optional URL allowlist for enterprise |
| Token burn on cancel | Cancel panel/judge tasks on client disconnect | Integration test for cancellation |
| Judge context overflow | Truncate longest panel answers first (`max_panel_tokens`) | Tokenizer-accurate counting |
| Concurrency / DoS | `limits.py` enforces an optional `max_in_flight` cap (`OverloadedError`/503) and a per-key `rate_limit_per_minute` window (`RateLimitError`/429); both off (unlimited) by default. The rate-limit key is only trusted from an *authenticated* Bearer token (checked against `gateway.api_keys`) -- without an allowlist configured, all traffic shares one `anonymous` bucket so a client can't bypass the limit by rotating headers (`server.py::_rate_limit_key`) | Limits are in-process/best-effort, not a substitute for an edge proxy or a distributed limiter |

## Testing layers

- **Unit** — config, panel, synthesize, stream (no network)
- **Integration** — FastAPI test client + `respx` mock upstream
- **Manual** — OpenAI Python SDK against a live server with real API keys

## Distribution

- `openfusion` console script → `cli.py`: bare command is the interactive chat REPL; `openfusion web` runs the server via uvicorn; `ask`/`setup` are one-shot/wizard
- Docker image mounts config via volume or env-substituted YAML
- CI runs `ruff check` and `pytest` without live API keys
