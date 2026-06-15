# openfusion Architecture

This document explains module boundaries, extension points, and security decisions for openfusion.

## Why these modules exist

openfusion is a thin FastAPI proxy. Each module owns one concern so strategies, routers, and eval harnesses can grow without rewrites.

| Module | Responsibility | Must NOT do |
|--------|----------------|-------------|
| `server.py` | HTTP routes, auth gate, routing (`openfusion` vs pass-through), cancellation orchestration | SSE framing, judge prompt logic |
| `config.py` | Typed config from YAML + env | HTTP or upstream calls |
| `cost.py` | Token ceilings and request cost policy | Provider-specific pricing math |
| `upstream.py` | Shared httpx client for OpenAI-compatible APIs | Business logic about panels or judges |
| `panel.py` | Parallel fan-out, timeouts, degrade, 429 retry | SSE framing |
| `synthesize.py` | Judge prompt assembly, yield text deltas only | SSE framing |
| `stream.py` | All OpenAI chunk/SSE framing, progress events, terminal usage | Judge prompt content decisions |
| `metrics.py` | In-process counters/latency/token+cost registry, Prometheus text rendering | HTTP, upstream calls, prompt/secret handling |

## Request flow

```
Client → server.py
           ├─ model != openfusion → upstream.py (pass-through)
           ├─ tools present → upstream.py (pass-through, no fusion)
           └─ model == openfusion
                 → panel.gather_panel()
                 → stream.synthesize_and_stream()
                       → synthesize.synthesize() (deltas)
                       → stream wraps deltas into SSE
```

## Extension points

1. **Synthesis strategies** — implement a new function in `synthesize.py` (or a `strategies/` package later) and select via `strategy` in config.
2. **Router gate** — add a pre-panel classifier in `server.py` before `gather_panel`; keep it optional for MVP.
3. **Eval harness** — `bench/` calls the same HTTP surface as production clients; no special internal APIs.

## Configuration and secrets

- Runtime config lives in `openfusion.yaml` (gitignored). Use `openfusion.yaml.example` as the template.
- `openfusion.dev.yaml.example` is the low-cost live-test recipe; it is intentionally smaller than
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
| Concurrency / DoS | No global in-flight cap in MVP; rely on httpx pool + uvicorn workers | Add semaphore + queue limits |

## Testing layers

- **Unit** — config, panel, synthesize, stream (no network)
- **Integration** — FastAPI test client + `respx` mock upstream
- **Manual** — OpenAI Python SDK against a live server with real API keys

## Distribution

- `openfusion` console script → `cli.py` → uvicorn
- Docker image mounts config via volume or env-substituted YAML
- CI runs `ruff check` and `pytest` without live API keys
