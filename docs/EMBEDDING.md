# Embedding openfusion

openfusion is built to be wrapped. A host application (auth, billing, persistence,
multi-tenancy) can reuse the whole engine — the `/v1` surface, the playground, the
fusion pipeline — and add only its own concerns on top.

## Per-request config (`config_resolver`)

By default `create_app()` serves one app-wide `OpenFusionConfig`. For a hosted /
multi-tenant deployment, pass a **resolver** that returns a config *per request* —
typically the authenticated user's recipe and their own (BYO) API key:

```python
from fastapi import Request
from openfusion.server import create_app
from openfusion.config import OpenFusionConfig

async def resolve_config(request: Request) -> OpenFusionConfig:
    user = await authenticate(request)          # your auth
    return load_user_config(user)               # their key + chosen panel/judge

app = create_app(config_resolver=resolve_config)  # no static config required
```

Every route (`/v1/chat/completions`, `/v1/config`, `/v1/models`) then uses the
resolved config. The resolver runs once per request, so it can be per-user,
per-API-key, or per-tenant. Provider keys live in the config the resolver returns
and never come from the client.

## Wrapping the app

Mount openfusion under your own FastAPI app and add auth/metering middleware,
account routes, a database, and a marketing site around it:

```python
from fastapi import FastAPI
host = FastAPI()
host.add_middleware(YourAuthMiddleware)
host.mount("/", create_app(config_resolver=resolve_config))
```

## Usage & cost

Per-request usage and cost are already exposed: the streaming response emits an
`event: usage` SSE payload, and non-streaming responses include a `usage` field
(summed across the panel and judge). A host can meter from those today. A
dedicated `usage_callback` hook is on the roadmap.

## Programmatic use (no HTTP)

The pipeline is importable for non-HTTP use: `panel.gather_panel`,
`synthesize.synthesize` / `stream.buffer_synthesis`, `vote.majority_vote`,
`ranked.pick_best`. See `openfusion/cli.py` (`_run_ask`) for a minimal example.

## Boundary

Keep auth dashboards, billing, abuse controls, and secret storage in the host
application — not in this repo. openfusion stays the open, self-hostable engine.
