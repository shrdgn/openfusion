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

## Next
- **Usage callback** — a `usage_callback` hook for per-request metering (usage is
  already exposed via the SSE `usage` event / response `usage` field).
- **Cost preview + response cache** — estimate spend before running; dedupe
  identical prompts.
- **Smarter routing** — embedding/classifier-based model selection beyond the
  current heuristic; learn from outcomes.

## Benchmarks
- Scale the DRACO eval to the full task set with a stronger grader for a
  quotable, rubric-graded number. See `bench/FINDINGS.md`.

## Hosted version
A hosted product (accounts, saved keys, history) is planned as a **separate
application** that depends on this package — not a fork. This repo stays the
open, self-hostable engine + playground.
