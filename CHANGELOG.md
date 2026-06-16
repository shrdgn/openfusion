# Changelog

All notable changes to openfusion are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `preset: quality | budget` config switch — expands to a diverse OpenRouter
  panel + judge with web search/fetch enabled by default (the regime where
  synthesis beats the best single member per `bench/FINDINGS.md`). Mirrors
  OpenRouter Fusion's Quality/Budget UX. Explicit YAML always overrides a preset.
- `openfusion.preset.yaml.example` — one-line recipe to copy.
- "unique insights" added to the judge synthesis prompt, matching Fusion's
  synthesis dimensions (consensus / contradictions / partial coverage /
  unique insights / blind spots).
- Startup summary banner (recipe, panel, judge, tools, listen address, and the
  model name to call) printed by the CLI.
- Open-source project files: `CONTRIBUTING.md`, `SECURITY.md`,
  `CODE_OF_CONDUCT.md`, issue forms, PR template, Dependabot, `CITATION.cff`.
- README: CI/license/Python badges and an openfusion-vs-Fusion comparison table.

### Changed
- Friendlier configuration errors: a missing config file suggests copying an
  example, and a missing `${ENV_VAR}` suggests how to export it. The CLI prints
  a clean message and exits non-zero instead of dumping a traceback.

### Fixed
- `--config` is now honored by the uvicorn factory worker, not just the
  `OPENFUSION_CONFIG` environment variable.

## [0.1.0]

Initial MVP: OpenAI-compatible `POST /v1/chat/completions` + `/v1/models`,
parallel panel fan-out with graceful degrade, judge synthesis and majority-vote
aggregators, real SSE streaming with progress events, pass-through for
non-fusion models and tool calls, configurable cost ceilings, Prometheus
metrics at `/metrics`, agentic web tools for panel members, and a reproducible
`bench/` harness.

[Unreleased]: https://github.com/shahar-dagan/openfusion/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/shahar-dagan/openfusion/releases/tag/v0.1.0
