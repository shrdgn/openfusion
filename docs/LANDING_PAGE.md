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

## Separation rules

- Keep the page static: HTML and global CSS only.
- Do not add client-side demos that call provider APIs directly.
- Keep technical source-of-truth details in `README.md`, `DESIGN.md`, and `docs/ARCHITECTURE.md`.
- Treat hosted-product features such as auth dashboards, billing, analytics, and abuse controls as a
  separate application boundary when they become real.

## Security concerns to keep reviewing

- Provider keys must never appear in landing page markup, CSS, JavaScript, examples, or browser logs.
- Hosted demos must use gateway auth, budget controls, and rate limits before accepting public traffic.
- External analytics or embeds should be avoided until there is a privacy policy and consent model.
- Marketing claims should stay tied to benchmark results that can be reproduced from `bench/`.
- Static assets should avoid prompt or response captures from real users.
