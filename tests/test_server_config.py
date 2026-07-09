"""Coverage for server endpoints/branches not tied to a specific feature test file:
the routing-outcomes observability endpoint and the no-config defensive guard.
"""

from __future__ import annotations

import httpx

from openfusion.config import GatewayAuthConfig, OpenFusionConfig, PanelMember
from openfusion.server import create_app


def _config() -> OpenFusionConfig:
    return OpenFusionConfig(
        panel=[PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m1")],
    )


async def test_routing_outcomes_endpoint_returns_snapshot() -> None:
    app = create_app(_config())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/v1/routing/outcomes")
    await app.state.upstream_client.aclose()
    assert res.status_code == 200
    assert "outcomes" in res.json()


async def test_routing_outcomes_endpoint_requires_gateway_auth() -> None:
    cfg = _config()
    cfg.gateway = GatewayAuthConfig(api_keys=["secret"])
    app = create_app(cfg)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        no_auth = await client.get("/v1/routing/outcomes")
        ok = await client.get(
            "/v1/routing/outcomes", headers={"Authorization": "Bearer secret"}
        )
    await app.state.upstream_client.aclose()
    assert no_auth.status_code == 401
    assert ok.status_code == 200


async def test_get_config_raises_when_no_config_and_no_resolver() -> None:
    """Defensive guard: an app with neither a static config nor a resolver
    can't serve any config-dependent route."""
    app = create_app(_config())
    app.state.config = None
    app.state.config_resolver = None
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/v1/config")
    await app.state.upstream_client.aclose()
    assert res.status_code == 502
    assert res.json()["error"]["message"] == "No configuration available"
