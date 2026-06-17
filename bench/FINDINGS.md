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
| Web-dependent, **tools on both sides** | diverse panel w/ web_search+fetch | **best** member (deepseek-v4-pro, also w/ tools) | **79%** | 52–92% | ~2× |

### DRACO (rubric-graded, the credible benchmark)

The pairwise rows above use an LLM judge picking a winner. DRACO instead grades
each answer against its ~40 weighted criteria (errors carry negative weight) and
reports a normalized 0–100 score — the methodology behind OpenRouter's result.
Subset of 10 tasks, tools (`web_search`+`web_fetch`) identical on both sides,
`gemini-3-flash` grader, 1 grading pass, 2048-token answers, 0 errors:

| System | Mean DRACO score | Per-domain (solo→fusion) |
|--------|-----------------|--------------------------|
| Solo (deepseek-v4-pro, tools) | **17.2%** | Medicine 48.7, Shopping 23.9, Tech 26.0, Finance 0, Academic 0, Personalized 0 |
| Fusion (budget panel, tools) | **42.1%** (**+24.9**) | Medicine 85.9, Shopping 59.4, Tech 32.3, Finance 46.1, Academic 0, Personalized 0 |

Fusion scored ≥ solo on **10/10** tasks (won 6, tied 4, lost 0). Absolute scores
run well below OpenRouter's (~60s) because of the 2048-token answer cap, a single
grading pass, and a flash-tier grader; the **relative gap is the signal**, and
DRACO's authors note system rankings are stable across judge/format choices.

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
5. **The lift requires tools — and with tools, synthesis clearly wins.** This
   is the headline reversal. When *both* sides get the same agentic tools
   (`openrouter:web_search` + `web_fetch`), the diverse panel beat its own best
   member **79%** (11–3, CI 52–92%) on web-dependent questions — the first
   decisive fusion win in the whole arc. The tool-free "synthesis is worthless"
   results (#3–4) were a **regime artifact**: without tools every panelist
   regurgitates the same parametric answer, so there is nothing to fuse; with
   tools they take different search/fetch trajectories → complementary evidence
   → the synthesis step has real material to work with. This matches OpenRouter:
   even *self*-fusion (same model twice) lifts +6.7 pts on DRACO *with* tools,
   vs. no lift in our tool-free self-fusion test.

## Implication for the roadmap

- **Tool-enabled fusion works — it shipped and is validated.** Set
  `tools.web_search: true` (+ `web_fetch`) to give panel members the agentic
  `openrouter:web_search`/`web_fetch` server tools; see
  `examples/panel-tools.yaml.example`. The corrected gate (tools on both
  sides) confirmed the lift. Run it with `run-bench/research-paneltools`.
- **Next: DRACO for the quotable number.** The gate is small-N (14 decided,
  wide CI) and uses a pairwise judge that can't verify currency. A credible
  "matches frontier for less" claim needs the rubric-graded benchmark — DRACO
  is public (`hf.co/datasets/perplexity-ai/draco`; grader:
  `github.com/The-LLM-Data-Company/rubric`), tools already identical across
  configs, and `excluded_domains` is wired for rubric-contamination control.
  Ideally add a frontier solo baseline (not just the best budget member).
  *(Done at subset scale: `run-bench/draco` rubric-graded 10 tasks, fusion
  42.1% vs solo 17.2%, +24.9, 0 errors — see the DRACO table above.)*
- **To harden the DRACO number:** scale to the full 100 tasks, raise the answer
  cap (2048 truncates deep-research reports — likely the main reason absolute
  scores trail OpenRouter's), and grade 3× with a stronger judge
  (gemini-3-pro / sonnet-4.6) per the paper.

## Caveats

- N≤40 per run; single judge (rankings are stable across judges per the DRACO
  paper, but absolute magnitudes shift 10–25 pts).
- Tool-free, English, single-turn analytical questions — not deep research.
- OpenRouter synthesized with Opus 4.8; our strongest synthesizer was
  `deepseek-v4-pro`. A frontier synthesizer might behave differently, but the
  direction (synthesis dilutes on tool-free tasks) was consistent across both
  synthesizers we tried.
