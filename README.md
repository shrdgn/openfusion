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

| Recipe | Tasks | Accuracy | Notes |
|--------|-------|----------|-------|
| Solo model | 20 | baseline | Same model, single sample |
| Self-fusion (N=3) | 20 | +lift on reasoning tasks | See `bench/README.md` for reproduction |

> Claim only what your local `bench/run.py` run proves on your chosen model and task set.

## Stack

Python 3.11+ / FastAPI / httpx / uvicorn.

## Landing page

The service serves a static project landing page from `GET /`. See
[docs/LANDING_PAGE.md](docs/LANDING_PAGE.md) for the repo-local website decision, migration trigger,
and security concerns to revisit before a hosted product accepts public traffic.

## License

MIT.
