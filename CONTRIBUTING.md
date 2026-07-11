# Contributing to openfusion

Thanks for your interest in openfusion. It's a thin, forkable proxy — the whole
point is that the recipe is yours to tune. Contributions that keep it small,
honest, and easy to run are very welcome.

## Ground rules

- **Keep modules honest.** Each module in `openfusion/` owns one concern and has
  a documented "must NOT do" in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
  Respect those boundaries (e.g. only `stream.py` does SSE framing; only
  `synthesize.py` builds the judge prompt).
- **Claim only what the bench proves.** openfusion makes no "beats frontier"
  claim. If a change affects quality or cost, back it with a `bench/run.py`
  number on stated models/tasks. See [`bench/FINDINGS.md`](bench/FINDINGS.md).
- **Never read provider keys from client requests.** Upstream API keys come from
  config/env only. See [SECURITY.md](SECURITY.md).

## Development setup

Requires Python 3.11+.

```bash
git clone https://github.com/shahar-dagan/openfusion
cd openfusion
pip install -e ".[dev]"
```

> Not yet published to PyPI — install from source. `uvx openfusion` / PyPI is a
> planned fast-follow (see [DESIGN.md](DESIGN.md)).

## Before you open a PR

CI runs two jobs — run the relevant checks locally first:

```bash
ruff check .          # lint + import order (ruff format-compatible)
pytest -q             # 280+ tests, no network (upstreams are mocked with respx)
```

If you touched `web/`, also run the UI job's checks:

```bash
cd web
npm ci
npm test              # Vitest
npm run build         # rebuilds openfusion/static/playground/
```

CI fails if the built UI in `openfusion/static/playground` isn't committed and
in sync with `web/`, so commit the rebuilt output alongside any `web/` change.

Tests must not require live API keys or network access. Mock upstreams with
`respx` (see `tests/test_integration.py` for the pattern). The only live path is
the opt-in `scripts/openrouter_smoke.py`, which never runs in CI.

## Pull request checklist

- [ ] `ruff check .` and `pytest -q` pass locally.
- [ ] New behavior has a test (unit or `respx`-mocked integration).
- [ ] Docs updated if you changed config, the request surface, or defaults
      (`README.md`, and `docs/ARCHITECTURE.md` for module boundaries).
- [ ] Quality/cost claims are backed by a reproducible bench number.
- [ ] No secrets, prompts, or response bodies added to logs or metrics.

## Good first contributions

Areas where help is especially useful (see "Open questions" / "Follow-up" in
`DESIGN.md` and `docs/ARCHITECTURE.md`, and "Next" in `ROADMAP.md`):

- Fusion-aware tool/function-calling (today tool requests pass through unfused).
- Live `$` pricing in `/v1/estimate` for non-OpenRouter providers, and a CLI
  cost preview (`openfusion ask --estimate`).
- Scaling the DRACO benchmark to the full task set with a stronger grader (see
  `bench/FINDINGS.md`).

## License

By contributing, you agree your contributions are licensed under the
[MIT License](LICENSE).
