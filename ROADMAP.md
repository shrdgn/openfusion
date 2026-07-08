# Roadmap

Directional, not a promise. Issues and PRs welcome on any of these.

## Done
- Core fusion (panel → judge), OpenAI-compatible API, SSE streaming.
- CLI chat REPL + `ask`, web playground, presets, debate/vote/ranked aggregators.
- Production limits (concurrency + rate limiting), prompt caching, web-tool fusion.
- **Model routing** — route each prompt to the best single model by difficulty
  (`router.route_models`), or fuse. See `examples/route.yaml.example`.
- **Side-by-side panel view** — each model's answer shown in the playground next
  to the fused result (`openfusion.expose_panel` → `panel_answer` SSE events).
- **Embeddable engine** — `create_app(config_resolver=...)` for per-request
  (multi-tenant) config; see `docs/EMBEDDING.md`. Foundation for a hosted app.
- **Response cache** (`response_cache.enabled`) — identical fused requests served
  from memory; `create_app(usage_callback=...)` meters usage per request.
- **Cost preview** — `POST /v1/estimate` (calls, tokens, and `$` from OpenRouter
  pricing); shown in the playground as you type.
- **Classifier model routing** — `mode: model` + `route_models` lets one
  classifier call pick FUSE or the specific model (`router.route_request`).
- **Routing that learns from outcomes** — an in-process EMA of fuse/solo success
  rate per prompt tier (`openfusion.outcomes.OutcomeStore`) nudges the heuristic
  router once either side has enough observations; read-only at
  `GET /v1/routing/outcomes`.

## Next
- **Live `$` pricing in the estimate for non-OpenRouter providers** and a
  CLI cost preview (`openfusion ask --estimate`).

## Benchmarks
- Scale the DRACO eval to the full task set with a stronger grader for a
  quotable, rubric-graded number. See `bench/FINDINGS.md`.

## Hosted version
A hosted product (accounts, saved keys, history) is planned as a **separate
application** that depends on this package — not a fork. This repo stays the
open, self-hostable engine + playground.
