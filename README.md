# openfusion

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
# Install from source
pip install -e .

# Copy and configure
cp openfusion.yaml.example openfusion.yaml
export OPENROUTER_API_KEY=your-key-here

# Run the server
openfusion --host 0.0.0.0 --port 8000
```

For credit-conscious local testing, start from `openfusion.dev.yaml.example`. It uses a smaller
two-sample recipe, a lower-cost OpenRouter model, and strict token ceilings.

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

## Parameter precedence

| Parameter | Applies to | Notes |
|-----------|------------|-------|
| `temperature` (client) | Judge only indirectly via recipe | Self-fusion varies panel temps from config, not client |
| `max_tokens`, `stop`, `response_format` | Judge (visible output) | Panel members use recipe defaults |
| `stream`, `stream_options` | Judge path | Panel always runs non-streamed internally |
| `tools` / `tool_calls` | Pass-through only | Tool requests skip fusion in MVP |

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

## License

MIT.
