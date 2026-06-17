# openfusion — design

Status: APPROVED · 2026-06-15

## Problem

OpenRouter's Fusion is a closed "compound model": it fans a prompt out to a panel of LLMs and
has a judge model synthesize one answer that beats any single frontier model (their report:
Fable-level quality at ~half the cost on the DRACO eval). Great idea, proprietary API.

openfusion is the open version: a drop-in, OpenAI-compatible proxy that gives any existing tool
mixture-of-agents quality from models the user already pays for, as a tunable recipe.

## Premises

1. **Surface** — drop-in OpenAI-compatible proxy: `POST /v1/chat/completions` + `GET /v1/models`.
   `model: "openfusion"` triggers the panel; other model names pass through untouched.
2. **Synthesis** — mixture-of-agents: parallel fan-out to a fixed panel, one judge model reads all
   responses (consensus / contradictions / partial coverage / blind spots) and writes the final
   answer. No per-prompt router in the MVP.
3. **Models** — each panel member + judge is `{base_url, api_key, model}`; OpenRouter is the default
   upstream but never a hard dependency (OpenAI / Together / local vLLM / Ollama all work).
4. **Streaming** — wait for the panel, then stream the judge token-by-token; emit cosmetic progress
   events while waiting so the client never sees a dead spinner. Real SSE.
5. **Config-driven** — panel, judge, strategy, timeouts in `openfusion.yaml`; the recipe is tunable.
6. **Distribution is part of the product** — `pip install` / `uvx openfusion` + a Docker image + a
   reproducible benchmark number, or it's code nobody runs.

## Recommended approach: thin proxy + self-fusion default

Build a thin FastAPI proxy as the spine. Ship **self-fusion** as the default recipe (sample ONE
model N times with varied temperature/seed/system, then judge) so the first demo runs on a single
API key. Implement synthesis as one swappable function so alternate strategies (vote, debate), a
router gate, and an eval harness grow out of it later without a rewrite.

### Modules

- `server.py` — FastAPI app: `/v1/chat/completions`, `/v1/models`, `/healthz`.
- `config.py` — load `openfusion.yaml` (panel list, judge, strategy, N, timeouts) + env keys.
- `panel.py` — `gather_panel(request)`: fan out members concurrently with per-member timeout; drop
  failed/timed-out members, proceed if ≥1 survives (degrade, don't fail the request).
- `synthesize.py` — `synthesize(...) -> AsyncIterator[str]`: build the judge prompt from member
  outputs and yield judge **text deltas only**. The single swap point for future strategies. Does
  NOT touch SSE framing.
- `stream.py` — owns ALL OpenAI-chunk/SSE framing: wraps judge deltas into chunks with one
  consistent `id`/`created`/`model`, emits progress events, writes the terminal `finish_reason` +
  final usage event + `[DONE]`.

### Default judge prompt (tunable starting point)

> You are the synthesizer. Below are N independent answers to the same user request. Identify points
> of consensus, contradictions, partial coverage, and blind spots. Then write a single best answer
> grounded in that analysis. Do not mention the panel or that multiple answers existed.

Panel responses injected as labeled blocks.

## Protocol & edge-case decisions

The edges where an OpenAI-compatible proxy lives or dies. Decided up front.

- **Auth boundary.** The client's `Authorization: Bearer` is an *openfusion gateway token* (optional
  local allowlist via `OPENFUSION_API_KEYS`; if unset, accept any). Upstream provider keys come from
  config/env only and are NEVER taken from the client.
- **Client error envelope.** All failures return the OpenAI shape
  `{"error": {"message", "type", "code"}}` with correct status: bad request → 400
  (`invalid_request_error`); all panel members down → 502 (`upstream_error`); judge fails before any
  token → 502; judge fails mid-stream → emit an SSE error data chunk then `[DONE]` (can't change the
  already-sent 200).
- **Progress-event encoding.** Progress is COSMETIC and never corrupts the answer body. Emit as
  custom SSE `event: progress` lines with JSON `data`. Strict OpenAI clients ignore non-`message`
  events and may just show a spinner — accepted. Never inject progress into `choices[].delta.content`.
- **Client param precedence.** The recipe owns panel sampling (self-fusion *varies* temp/seed, so a
  client `temperature` does NOT reach panel members). Client `max_tokens`, `stop`, `response_format`
  apply to the **judge** (the visible output). Documented in the README.
- **Cost controls.** Configured token ceilings fill missing `max_tokens` values. Pass-through and
  judge calls reject client values above the ceiling; panel calls clamp internally because panel
  output is intermediate and should not let one large client value multiply across N calls.
- **Usage is best-effort.** Request `stream_options: {include_usage: true}` from the judge upstream;
  sum non-streamed panel usage when present. If an upstream omits usage, prefer omitting the field;
  a local tokenizer estimate is allowed only when clearly flagged as an estimate. Per-member
  breakdown ships as a final SSE `event: usage` payload, NOT response headers (headers flush before
  panel usage is known).
- **Cancellation.** On client disconnect (`await request.is_disconnected()` / task cancellation),
  cancel outstanding panel + judge tasks. The "half the cost" pitch depends on not burning tokens
  after the user hits stop.
- **Timeouts.** Per-member timeout + a separate judge timeout + a total wall-clock budget.
- **Judge input budget.** Cap injected panel-answer tokens (truncate longest-first) so N long
  answers + the original prompt can't blow the judge's context window.
- **Concurrency (MVP non-goal, stated).** No global in-flight cap; rely on httpx pool limits + uvicorn
  workers. Documented as a known limitation.
- **Self-fusion 429.** N identical-model calls can rate-limit together. Retry with backoff on 429 for
  the self-fusion default; otherwise degrade (proceed if ≥1 survives). The retry loop honors the
  disconnect + total-budget check, so a backoff sleep never keeps burning time after the client left.

## Success criteria

- `uvx openfusion` boots a server; the OpenAI Python SDK pointed at it with `model="openfusion"`
  returns a streamed, synthesized answer.
- Works unmodified as a custom OpenAI endpoint in at least one real tool (Cursor or Cline).
- A `bench/` script reproduces a head-to-head number: self-fusion recipe vs. the same model solo on a
  small public eval (50–100 tasks), showing a measurable lift.
- A panel member timing out or erroring degrades gracefully rather than 500-ing the request.

## Distribution

- **Weekend bar:** `pip install .` from source + local `docker build`/`docker run`.
- **Fast-follow:** PyPI (`uvx openfusion`) + published GHCR image + GitHub Actions (lint/test on PR,
  tag push builds wheel + image).
- **Proof artifact:** `bench/` results + a short README table. The number is the pitch.

## Open questions

- **Tool/function calling** — *partially resolved.* Server-executable web tools
  (`openrouter:web_search`/`web_fetch`) now fuse: panel members run them upstream and the judge
  synthesizes the results. Client-side function tools and mid-conversation tool turns still pass
  through to a single model, since their results return through the client and can't be fused into
  one answer without running each panelist's tool loop.
- **Prompt-caching** for self-fusion's shared prefix — defer, measure first.
- **Non-streaming requests** — support `stream: false` by buffering the judge (trivial, do it).
- **"Beats frontier" honesty** — the in-repo bar is "a measurable lift on a small eval," not
  "beats Fable." Claim only what `bench/` shows.
- **Provider budgets** — token ceilings reduce accidental spend but do not replace provider-side
  budgets, per-key rate limits, or alerting.

## Build order

1. Scaffold: `pyproject.toml`, `openfusion/` package, `openfusion.yaml.example`, ruff + pytest.
2. `/v1/chat/completions` happy path: parse OpenAI request, fan out a 1-member panel, pass through.
3. `gather_panel` concurrency + per-member timeout + graceful degrade.
4. `synthesize` judge + SSE streaming with progress events.
5. Self-fusion as the default `openfusion.yaml` (one model, N=3).
6. `bench/` head-to-head; put the number in the README.
7. Dockerfile + GitHub Actions + PyPI publish (fast-follow).
