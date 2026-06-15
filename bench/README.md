# openfusion benchmark

Head-to-head comparison of **self-fusion** (panel + judge) vs a **solo** call to the same model.

> **What we learned running these:** see [FINDINGS.md](FINDINGS.md) — fusion only
> pays off in the right regime, and on tool-free tasks synthesis adds no value
> over the best panel member. Tools are the next lever.

## Usage

```bash
export OPENROUTER_API_KEY=your-key
python bench/run.py \
  --config openfusion.dev.yaml.example \
  --tasks bench/tasks/smoke.jsonl \
  --max-tokens 32 \
  --output bench/results/latest.json
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `openfusion.yaml.example` | openfusion config with panel/judge |
| `--tasks` | `bench/tasks/sample.jsonl` | JSONL task file |
| `--dataset` | none | Public dataset loader (`gsm8k`); overrides `--tasks` |
| `--limit` | `40` | Max tasks when using `--dataset` |
| `--base-url` | `http://127.0.0.1:8000/v1` | openfusion proxy URL |
| `--solo-model` | from config panel | Model name for solo baseline |
| `--output` | stdout | Write JSON results to file |
| `--max-tokens` | `64` | Per-call output cap for both solo and fusion requests |
| `--fail-on-regression` | off | Exit non-zero if fusion accuracy < solo (off by default) |

## Quality per dollar

The point of fusion is rarely "win on an easy set" — it is reaching a quality bar more cheaply, or
adding reliability where a single call is shaky. So each mode reports not just `accuracy` but
`total_cost_usd`, `total_tokens`, **`cost_per_correct_usd`**, and `tokens_per_correct`. Compare
`cost_per_correct_usd` across modes to see whether fanning out actually buys anything.

A completed run exits 0 even if fusion does not beat solo — the table is the result, not a
pass/fail. Use `--fail-on-regression` if you want CI to go red on an accuracy regression.

## Harder eval (GSM8K)

Easy sets saturate (solo already ~100%), leaving no room for fusion to help. For a reasoning set
with headroom, use the GSM8K loader with the higher-ceiling bench config:

```bash
python bench/run.py \
  --config openfusion.bench.yaml.example \
  --dataset gsm8k --limit 40 \
  --max-tokens 512 \
  --output bench/results/latest.json
```

GSM8K answers are graded with the `numeric` match mode (the final number in the response is
compared to the gold answer). The test split is fetched once from the public
`openai/grade-school-math` repo and cached under `bench/.cache/` (gitignored).

## Open-ended (research) eval

Deep-research tasks are open-ended, so exact-match scoring does not apply.
`bench/research_eval.py` instead gets a solo answer and a fusion answer for each
question, then asks a judge model which is better in a **blind, position-randomized
A/B comparison** (the judge is told not to reward verbosity). It reports fusion
wins / solo wins / ties and the cost per mode — a cheap proof of whether the
fusion synthesis step lifts open-ended quality, before investing in tool-enabled
fusion and a full rubric-graded harness.

```bash
python bench/research_eval.py \
  --config openfusion.bench.yaml.example \
  --tasks bench/tasks/research.jsonl \
  --max-tokens 512 \
  --output bench/results/latest.json
```

Caveat: same-family judge (gpt-4o-mini grading gpt-4o-mini) has some self/length
bias; position randomization and the no-verbosity instruction mitigate but do not
eliminate it. This is a directional proof, not a rubric-grade benchmark.

## Task format

Each line in a `--tasks` JSONL file:

```json
{"id": "q1", "prompt": "...", "expected": "42", "match": "exact"}
```

Supported `match` modes:

- `exact` — normalized string equality
- `contains` — expected substring appears in answer
- `numeric` — the last number in the answer equals the expected number (used by GSM8K)

## Reproducing README numbers

1. Start openfusion: `OPENFUSION_CONFIG=openfusion.dev.yaml.example openfusion --port 8000`
2. Run the smoke task set with `--max-tokens 32`
3. Move to `bench/tasks/sample.jsonl` only after the smoke run is stable
4. Compare `solo.accuracy` vs `fusion.accuracy` in the output JSON

Each result also carries `total_tokens` and `total_cost_usd` per mode, so you can weigh any
accuracy lift against the extra spend of fanning out to a panel.

The sample set is small (20 tasks) and intended as a smoke benchmark, not a frontier eval.

## Running in CI

The `Benchmark` GitHub Actions workflow (`.github/workflows/bench.yml`) runs this head-to-head
against the repo's `OPENROUTER_API_KEY` secret, so the key never touches a local session. Because
it spends real credits, it runs only on manual dispatch (Actions tab → Benchmark → Run workflow)
or when a `run-bench/**` branch is pushed. The branch suffix selects a preset:

| Branch | Tasks | Config | max_tokens |
|--------|-------|--------|------------|
| `run-bench/smoke` | `smoke.jsonl` | dev | 32 |
| `run-bench/sample` | `sample.jsonl` | dev | 32 |
| `run-bench/gsm8k` | GSM8K (`--limit 40`) | bench | 512 |
| `run-bench/gsm8k-vote` | GSM8K, vote aggregator | bench-vote | 512 |
| `run-bench/research` | `research.jsonl` (pairwise judge) | bench | 512 |

`workflow_dispatch` inputs override any preset. Results are published as a job summary table
(accuracy, latency, tokens, cost, **$/correct**) and a `bench-results` artifact.

## Security

- Task files may contain sensitive prompts; do not commit private eval data.
- Results JSON may include model outputs; treat as confidential if prompts are.
- Benchmark loops spend credits quickly; use `bench/tasks/smoke.jsonl` and a low `--max-tokens`
  value before running larger task files.
