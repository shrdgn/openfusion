# openfusion benchmark

Head-to-head comparison of **self-fusion** (panel + judge) vs a **solo** call to the same model.

## Usage

```bash
export OPENROUTER_API_KEY=your-key
python bench/run.py \
  --config openfusion.yaml.example \
  --tasks bench/tasks/sample.jsonl \
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

## Task format

Each line in the JSONL file:

```json
{"id": "q1", "prompt": "...", "expected": "42", "match": "exact"}
```

Supported `match` modes:

- `exact` — normalized string equality
- `contains` — expected substring appears in answer

## Reproducing README numbers

1. Start openfusion: `openfusion --port 8000`
2. Run the benchmark against the sample task set
3. Compare `solo.accuracy` vs `fusion.accuracy` in the output JSON

The sample set is small (20 tasks) and intended as a smoke benchmark, not a frontier eval.

## Security

- Task files may contain sensitive prompts; do not commit private eval data.
- Results JSON may include model outputs; treat as confidential if prompts are.
