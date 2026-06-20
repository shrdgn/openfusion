# Roadmap

Directional, not a promise. Issues and PRs welcome on any of these.

## Done
- Core fusion (panel → judge), OpenAI-compatible API, SSE streaming.
- CLI chat REPL + `ask`, web playground, presets, debate/vote/ranked aggregators.
- Production limits (concurrency + rate limiting), prompt caching, web-tool fusion.
- **Model routing** — route each prompt to the best single model by difficulty
  (`router.route_models`), or fuse. See `examples/route.yaml.example`.

## Next
- **Side-by-side panel view** — stream each panel member's answer in parallel in
  the playground, then the fused result with the disagreement/blind-spot
  highlights. (Surfaces the "run multiple models side-by-side" promise; needs a
  per-member `panel_answer` SSE event + a columns UI.)
- **Embeddable engine** — expose the pipeline plus a per-request config resolver
  and a usage/cost callback, so the proxy can be wrapped (auth, metering) without
  forking. Foundation for a hosted version.
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
