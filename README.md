# openfusion

[![CI](https://github.com/shahar-dagan/openfusion/actions/workflows/ci.yml/badge.svg)](https://github.com/shahar-dagan/openfusion/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

An open-source, drop-in compound-model proxy. Point any OpenAI-compatible tool at it,
set `model: "openfusion"`, and your prompt is fanned out to a panel of LLMs in parallel —
then a judge model reads every response (consensus, contradictions, blind spots) and streams
back a single synthesized answer that aims to beat any one of them.

It's the open version of the mixture-of-agents idea behind OpenRouter's Fusion: better answers
from models you already pay for, as a tunable, forkable recipe instead of a black box.

## Status

**MVP** — self-fusion proxy with panel fan-out, judge synthesis, SSE streaming, and pass-through
for non-fusion models and tool calls. See [DESIGN.md](DESIGN.md) for architecture and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module boundaries and security notes.

## Quick start

```bash
# Install from source (not yet on PyPI)
pip install -e .

# Pick a one-line recipe and set your key
cp openfusion.preset.yaml.example openfusion.yaml   # contains: preset: budget
export OPENROUTER_API_KEY=your-key-here

# Run the server
openfusion --host 0.0.0.0 --port 8000
```

A **preset** is the whole recipe: `preset: quality` (or `budget`) expands to a diverse OpenRouter
panel plus a judge with web search/fetch enabled — the tool-enabled regime where fusion actually
beats the best single member (see [Benchmarks](#benchmarks)). This mirrors OpenRouter Fusion's
Quality/Budget switch. Anything you set explicitly in YAML overrides the preset.

| Preset | Panel | Judge | Tools |
|--------|-------|-------|-------|
| `quality` | Claude Sonnet 4 · Gemini 3 Pro · DeepSeek V4 Pro | Claude Sonnet 4 | web search + fetch |
| `budget` | GPT-4o-mini · DeepSeek V4 Pro · Kimi K2.6 | DeepSeek V4 Pro | web search + fetch |

Prefer to spell out every model, or run the cheaper self-fusion recipe on a single model? Start
from `openfusion.yaml.example` (full panel/judge config) or `openfusion.dev.yaml.example` (a smaller
two-sample, low-cost, strict-ceiling recipe for local testing) instead.

Use with the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local-dev")
stream = client.chat.completions.create(
    model="openfusion",
    messages=[{"role": "user", "content": "Explain mixture-of-agents in one paragraph."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

## Routing & strategies

Three knobs control *whether* and *how* a prompt is fused. All are optional and off/default.

- **Auto Router** (`router.enabled: true`) — a per-prompt gate that answers simple prompts with a
  single pass-through call and reserves the panel for prompts that look like they benefit (long,
  analytical, or containing code). Default is a cheap heuristic (no extra model call); `mode: model`
  uses a small classifier model and falls back to the heuristic if it errors:

  ```yaml
  router:
    enabled: true
    mode: heuristic     # heuristic | model | always | never
    min_chars: 280      # prompts at/over this length fuse
    # classifier:       # required for mode: model
    #   base_url: https://openrouter.ai/api/v1
    #   api_key: ${OPENROUTER_API_KEY}
    #   model: openai/gpt-4o-mini
  ```

- **Strategy** (`strategy:`) — how the panel is produced: `self_fusion` (one model sampled N times),
  `panel` (a fixed diverse panel), or `debate` (a diverse panel where each member revises after
  seeing the others' answers, then the judge synthesizes). Debate trades extra cost/latency for
  cross-examination:

  ```yaml
  strategy: debate
  debate:
    rounds: 1           # revision rounds before the judge
  ```

- **Aggregator** (`aggregator:`) — how answers become one: `judge` (synthesis, default), `vote`
  (majority vote, cheaper, best for verifiable short-answer tasks), or `ranked` (one short judge
  call picks the single best answer — cheaper than synthesis, uses model judgment unlike vote).

- **Analysis transparency** (`analysis.emit: true`) — surface the judge's structured reasoning
  (consensus / contradictions / partial coverage / unique insights / blind spots) as a separate SSE
  `event: analysis` (and an `analysis` field on non-streaming responses), without polluting the
  answer body.

- **Prompt caching** (`cache.enabled: true`) — mark the shared prefix so self-fusion's N samples
  reuse a cached prompt on providers that support it (a no-op elsewhere).

## Production limits

For public deployments, bound load and spend (both default to `0` = unlimited):

```yaml
limits:
  max_in_flight: 64           # cap concurrent requests; over-limit returns 503
  rate_limit_per_minute: 60   # per gateway key (or per client when unauthenticated); over-limit returns 429
```

These are best-effort, single-process guards — pair them with provider-side budgets and, for
multi-replica deployments, an edge rate limiter.

## How it works

```
client (Cursor / OpenAI SDK / anything)
        │  POST /v1/chat/completions   model="openfusion"
        ▼
   openfusion proxy ──► panel member A ┐
                   ──► panel member B ├─ parallel fan-out
                   ──► panel member C ┘
                        │
                        ▼
                   judge model  ──►  streamed synthesized answer (SSE)
```

- **Drop-in.** OpenAI-compatible `POST /v1/chat/completions` + `/v1/models`, real SSE streaming.
- **Default recipe is self-fusion.** Sample one model N times and judge the spread — works on a
  single API key without multi-provider juggling.
- **No lock-in.** Each panel member + judge is `{base_url, api_key, model}`. OpenRouter is the
  default upstream; OpenAI, Together, local vLLM/Ollama all work.
- **Config-driven.** Panel, judge, strategy, and timeouts live in `openfusion.yaml`.

## openfusion vs. OpenRouter Fusion

openfusion is the open implementation of the same idea. The core mechanism is at parity; the
differences are scale and a per-prompt router.

| | OpenRouter Fusion | openfusion |
|---|---|---|
| Parallel panel → judge synthesis | ✅ | ✅ |
| Synthesis dimensions | consensus · contradictions · partial coverage · unique insights · blind spots | same |
| Web search + fetch on the panel | ✅ (default) | ✅ (on by default with `preset:`) |
| Quality / Budget presets | ✅ | ✅ (`preset: quality \| budget`) |
| Override panel + judge | ✅ (plugin fields) | ✅ (any `{base_url, api_key, model}` in YAML) |
| Per-call cost breakdown | ✅ (Activity) | ✅ (SSE `usage` event + `/metrics`) |
| Self-hostable / forkable | ❌ closed API | ✅ MIT, any OpenAI-compatible provider |
| Per-prompt Auto Router | ✅ | ✅ heuristic or model classifier (`router.enabled`) |
| Structured analysis surfaced | ✅ | ✅ `analysis.emit` (SSE `analysis` event) |
| Multi-round debate | — | ✅ `strategy: debate` |
| Concurrency cap + rate limiting | ✅ | ✅ `limits` (best-effort, single-process) |
| Headline benchmark | full DRACO (100 tasks) | DRACO subset (10 tasks) — see [bench/FINDINGS.md](bench/FINDINGS.md) |

## Parameter precedence

| Parameter | Applies to | Notes |
|-----------|------------|-------|
| `temperature` (client) | Judge only indirectly via recipe | Self-fusion varies panel temps from config, not client |
| `max_tokens`, `stop`, `response_format` | Judge (visible output) | Panel members use recipe defaults |
| `stream`, `stream_options` | Judge path | Panel always runs non-streamed internally |
| `tools` / `tool_calls` | Fusion or pass-through | Server-executable web tools (`openrouter:web_search`/`web_fetch`) are fused; client-side function tools and mid-conversation tool turns pass through |

## Environment variables

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_API_KEY` | Default upstream key (via `${OPENROUTER_API_KEY}` in config) |
| `OPENFUSION_CONFIG` | Path to config file (default: `openfusion.yaml`) |
| `OPENFUSION_API_KEYS` | Comma-separated gateway allowlist (optional) |
| `OPENFUSION_HOST` / `OPENFUSION_PORT` | Server bind address |

## Cost safety and live smoke tests

`cost_controls` in config caps `max_tokens` for pass-through, panel, and judge calls. Missing
`max_tokens` values are filled from the configured ceiling; over-limit pass-through and judge
requests return `400`, while internal panel calls clamp to their ceiling.

Run the opt-in live OpenRouter smoke test only when you intend to spend a small number of credits:

```bash
export OPENROUTER_API_KEY=your-key
python scripts/openrouter_smoke.py --config openfusion.dev.yaml.example --yes-spend-credits
```

## Benchmarks

Run the head-to-head benchmark (self-fusion vs solo model):

```bash
pip install -e ".[dev]"
python bench/run.py --config openfusion.yaml.example --tasks bench/tasks/sample.jsonl
```

Use `--tasks bench/tasks/smoke.jsonl --max-tokens 32` before larger benchmark runs.

Each run reports accuracy **plus** the spend it took to get there — `total_tokens` and
`total_cost_usd` per mode — so you can weigh any accuracy change against the extra cost of fanning
out to a panel.

### What we measure today

The bundled `bench/tasks/sample.jsonl` (20 short Q&A tasks) is **saturated** for a capable model —
the solo baseline already scores ~100%, so there is no headroom for fusion to add accuracy. On a
recent run with `openai/gpt-4o-mini` (self-fusion N=2, `max_tokens=32`):

| Mode | Accuracy | Avg latency | Tokens | Cost |
|------|----------|-------------|--------|------|
| Solo | 100% (20/20) | 0.55s | 536 | $0.0001 |
| Self-fusion | 95% (19/20) | 1.40s | 4,669 | $0.0008 |

So on easy tasks fusion does **not** beat a single call — it costs more (here ~9× the tokens) and
can even regress, because the judge only has trivially-correct answers to choose between. This is
expected: mixture-of-agents helps where a single model is *unreliable*, not where it is already
right.

> openfusion makes **no** "beats frontier" claim. Demonstrating where fusion earns its cost needs
> a harder eval (one the solo baseline does not already ace) scored on **quality per dollar**, not
> accuracy alone. That eval is in progress; this table will be updated to show where fusion does
> and doesn't pay off. Claim only what your own `bench/run.py` run proves on your model and tasks.

## Observability

The proxy exposes Prometheus metrics at `GET /metrics` (no auth; scrape-only, bind accordingly):

- `openfusion_requests_total{route,outcome}` — client-facing requests (`fusion` / `pass_through`).
- `openfusion_upstream_requests_total{phase,outcome}` — upstream calls by `panel` / `judge` / `pass_through`.
- `openfusion_panel_members_total{outcome}` — per-member success vs. degraded failures.
- `openfusion_tokens_total{phase,kind}` and `openfusion_cost_usd_total{phase}` — token and cost spend.
- `openfusion_request_latency_ms` / `openfusion_upstream_latency_ms` — latency summaries (`_count` + `_sum`).

Cost (`usage.cost`, when the upstream reports it) is also rolled into the per-request SSE
`event: usage` payload and the non-streaming `usage` field, so a single fusion call shows what it
spent across the panel and judge. Per-call structured logs remain on the `openfusion.upstream`
logger.

## Stack

Python 3.11+ / FastAPI / httpx / uvicorn.

## Landing page

The service serves a static project landing page from `GET /`. See
[docs/LANDING_PAGE.md](docs/LANDING_PAGE.md) for the repo-local website decision, migration trigger,
and security concerns to revisit before a hosted product accepts public traffic.

## Contributing

Contributions are welcome — openfusion is meant to be forked and tuned. See
[CONTRIBUTING.md](CONTRIBUTING.md) for dev setup and the PR checklist, and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Please report security issues privately
per [SECURITY.md](SECURITY.md) rather than as a public issue.

## License

MIT.
