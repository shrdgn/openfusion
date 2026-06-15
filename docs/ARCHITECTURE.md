# openfusion Architecture

This document explains module boundaries, extension points, and security decisions for openfusion.

## Why these modules exist

openfusion is a thin FastAPI proxy. Each module owns one concern so strategies, routers, and eval harnesses can grow without rewrites.

| Module | Responsibility | Must NOT do |
|--------|----------------|-------------|
| `server.py` | HTTP routes, auth gate, routing (`openfusion` vs pass-through), cancellation orchestration | SSE framing, judge prompt logic |
| `config.py` | Typed config from YAML + env | HTTP or upstream calls |
| `upstream.py` | Shared httpx client for OpenAI-compatible APIs | Business logic about panels or judges |
| `panel.py` | Parallel fan-out, timeouts, degrade, 429 retry | SSE framing |
| `synthesize.py` | Judge prompt assembly, yield text deltas only | SSE framing |
| `stream.py` | All OpenAI chunk/SSE framing, progress events, terminal usage | Judge prompt content decisions |

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
- `${ENV_VAR}` placeholders in YAML are expanded at load time; missing env vars fail fast.
- Upstream provider API keys come from config/env only. Client `Authorization` is an optional openfusion gateway token.

## Security considerations

| Concern | Mitigation | Follow-up |
|---------|------------|-----------|
| Upstream key exfiltration | Never read provider keys from client headers or body | Audit logs for accidental key emission |
| Gateway auth bypass | Optional `OPENFUSION_API_KEYS` / `gateway.api_keys` allowlist | Rate limiting per gateway key |
| Secret logging | Redact `Authorization` and `api_key` fields in debug logs | Structured log scrubber |
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
