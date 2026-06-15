# openfusion benchmark

Head-to-head comparison of **self-fusion** (panel + judge) vs a **solo** call to the same model.

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
| `--base-url` | `http://127.0.0.1:8000/v1` | openfusion proxy URL |
| `--solo-model` | from config panel | Model name for solo baseline |
| `--output` | stdout | Write JSON results to file |
| `--max-tokens` | `64` | Per-call output cap for both solo and fusion requests |

## Task format

Each line in the JSONL file:

```json
{"id": "q1", "prompt": "...", "expected": "42", "match": "exact"}
```

Supported `match` modes:

- `exact` — normalized string equality
- `contains` — expected substring appears in answer

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
against the repo's `OPENROUTER_API_KEY` secret, so the key never touches a local session. It is
**manual dispatch only** (Actions tab → Benchmark → Run workflow) because it spends real credits.
Defaults are the cheap smoke set with `--max-tokens 32`; results are published as a job summary
table and a `bench-results` artifact.

## Security

- Task files may contain sensitive prompts; do not commit private eval data.
- Results JSON may include model outputs; treat as confidential if prompts are.
- Benchmark loops spend credits quickly; use `bench/tasks/smoke.jsonl` and a low `--max-tokens`
  value before running larger task files.
