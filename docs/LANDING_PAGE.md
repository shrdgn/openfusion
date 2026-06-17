# Landing page architecture

The landing page lives inside `openfusion/static/landing/` and is served by the FastAPI app at
`GET /`. This keeps the open-source project, install instructions, architecture proof points, and a
future hosted root together while openfusion is still an MVP.

## Why it is in this repo

- The project is currently a single Python service with no separate frontend workspace.
- A static page adds no runtime dependencies, build step, database, cache, or queue.
- The Docker image already copies the `openfusion/` package, so the page ships with local and hosted
  service deployments.
- The page can move to a separate website repository later without changing `/v1` API behavior.

## The playground is the one allowed interactive surface

`GET /playground` (in `openfusion/static/playground/`) is an interactive single-page demo. It is
permitted under these separation rules because it talks **only to the local `/v1` API of the same
openfusion server** — never to provider APIs, and it never embeds provider keys. The server holds
the keys; the browser at most sends an optional gateway token. Per-request model overrides are off
unless the operator sets `allow_request_overrides: true`, and even then are bounded by gateway auth,
cost ceilings, and rate limits. For a hosted deployment, keep overrides behind auth and rate limits.

## Separation rules

- Keep the marketing page (`/`) static: HTML and global CSS only.
- Do not add client-side demos that call provider APIs directly. (The `/playground` calls only the
  local `/v1` API, never a provider — see above.)
- Keep technical source-of-truth details in `README.md`, `DESIGN.md`, and `docs/ARCHITECTURE.md`.
- Treat hosted-product features such as auth dashboards, billing, analytics, and abuse controls as a
  separate application boundary when they become real.

## Security concerns to keep reviewing

- Provider keys must never appear in landing page markup, CSS, JavaScript, examples, or browser logs.
- Hosted demos must use gateway auth, budget controls, and rate limits before accepting public traffic.
- External analytics or embeds should be avoided until there is a privacy policy and consent model.
- Marketing claims should stay tied to benchmark results that can be reproduced from `bench/`.
- Static assets should avoid prompt or response captures from real users.
