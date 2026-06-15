# Findings: when does fusion beat solo?

A log of what the benchmarks actually showed, so we don't relearn it. All
numbers are from `gpt-4o-mini`-class budget panels against a live OpenRouter
key, graded as described below. Treat them as **directional** (N≤40, single
judge), not publication-grade.

## How we measured

- **Verifiable tasks** (math, short-answer): exact / numeric match, accuracy +
  cost-per-correct. See `bench/run.py`.
- **Open-ended tasks** (`bench/tasks/research.jsonl`, 40 analytical questions,
  no tools): there is no ground truth, so `bench/research_eval.py` gets a solo
  answer and a fusion answer and asks a **cross-family judge**
  (`google/gemini-3-flash-preview`) which is better, **blind and
  position-randomized**, instructed not to reward verbosity. We report the
  fusion win rate over *decided* comparisons with a **Wilson 95% CI**, plus
  `unparsed` (judge gave no A/B/TIE) and `errors` (skipped calls) so a broken
  run can't masquerade as a real result.

## Results

| Regime | Aggregator | Baseline | Fusion win rate | 95% CI | Cost vs solo |
|--------|-----------|----------|-----------------|--------|--------------|
| GSM8K-40 (verifiable) | judge | gpt-4o-mini | 87.5% vs 90% acc (worse) | — | ~5× |
| GSM8K-40 (verifiable) | vote | gpt-4o-mini | 87.5% (= judge) | — | ~2.5× (cheaper than judge) |
| Open-ended | self-fusion (judge) | gpt-4o-mini | 33% | 14–61% | ~5× |
| Open-ended | diverse panel (judge) | **weakest** member (gpt-4o-mini) | **97%** | 85–99% | ~23× |
| Open-ended | diverse panel, cheap synth | **best** member (deepseek-v4-pro) | 51% | 36–66% | ~2.5× |
| Open-ended | diverse panel, strong synth | **best** member (deepseek-v4-pro) | **29%** | 17–46% | ~4× |

## What it means

1. **Wrong regime kills fusion.** On saturated short-answer/math there is one
   right token and nothing to synthesize; the judge can only corrupt an answer
   the panel already had. Fusion ≤ solo on every verifiable set.
2. **Vote > judge for verifiable tasks.** Majority vote matched the judge's
   accuracy at ~40% lower cost (no synthesis call). Use `aggregator: vote` for
   short-answer; reserve the judge for open-ended.
3. **Diversity is just access to a better model.** A diverse panel crushed the
   *weakest* member (97%) — but that is because the panel *contained* stronger
   models, not because synthesis added anything.
4. **Synthesis adds no value on tool-free tasks — and can subtract it.**
   Measured against the panel's *best* member, fusion never wins: a cheap
   synthesizer ties (51%), and a **strong** synthesizer *loses* (29%) because
   reconciling weaker panelists' answers anchors and dilutes a model that would
   have answered better alone. All at 2.5–4× the cost.
5. **The lift requires tools.** OpenRouter's "beyond frontier" result (DRACO)
   gave every panelist `web_search`/`web_fetch`, so panelists gather
   *complementary evidence* worth synthesizing. Without tools, every panelist
   draws on the same parametric knowledge — there is nothing to fuse.

## Implication for the roadmap

- **Tool-enabled fusion is the next build** — panel members must run with web
  search/fetch instead of being passed through to a single model. Our data says
  it is the *only* untested lever that can make synthesis beat the best
  component. *(Capability landed: set `tools.web_search: true` to inject the
  upstream web plugin into panel calls — see `openfusion.panel-tools.yaml.example`.
  Still needs a web-dependent eval to demonstrate the lift.)*
- A credible "matches frontier for less" claim also needs a **frontier solo
  baseline** (not just the best budget member) and a **rubric-graded harness**
  (DRACO is public: `hf.co/datasets/perplexity-ai/draco`; grader:
  `github.com/The-LLM-Data-Company/rubric`).

## Caveats

- N≤40 per run; single judge (rankings are stable across judges per the DRACO
  paper, but absolute magnitudes shift 10–25 pts).
- Tool-free, English, single-turn analytical questions — not deep research.
- OpenRouter synthesized with Opus 4.8; our strongest synthesizer was
  `deepseek-v4-pro`. A frontier synthesizer might behave differently, but the
  direction (synthesis dilutes on tool-free tasks) was consistent across both
  synthesizers we tried.
