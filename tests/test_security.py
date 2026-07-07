"""Security tests: gateway auth, rate limits, and credential safety."""

from __future__ import annotations

import httpx
import pytest
import respx

from openfusion.config import (
    GatewayAuthConfig,
    JudgeConfig,
    LimitsConfig,
    OpenFusionConfig,
    PanelMember,
    PassThroughConfig,
    SelfFusionConfig,
    Strategy,
    TimeoutsConfig,
)
from openfusion.server import create_app


def _base_config(**overrides) -> OpenFusionConfig:
    return OpenFusionConfig(
        strategy=Strategy.SELF_FUSION,
        panel=[PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m")],
        judge=JudgeConfig(base_url="https://mock.upstream/v1", api_key="k", model="judge"),
        self_fusion=SelfFusionConfig(n=1),
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=15),
        pass_through=PassThroughConfig(
            base_url="https://mock.upstream/v1", api_key="k", model="pass-m"
        ),
        **overrides,
    )


@pytest.fixture
def gateway_config() -> OpenFusionConfig:
    return _base_config(gateway=GatewayAuthConfig(api_keys=["correct-key"]))


# ---------------------------------------------------------------------------
# /v1/chat/completions auth
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_completions_no_auth_returns_401(
    gateway_config: OpenFusionConfig,
) -> None:
    app = create_app(gateway_config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    await app.state.upstream_client.aclose()
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_chat_completions_wrong_key_returns_401(
    gateway_config: OpenFusionConfig,
) -> None:
    app = create_app(gateway_config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer wrong-key"},
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    await app.state.upstream_client.aclose()
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_chat_completions_correct_key_proceeds(
    gateway_config: OpenFusionConfig,
) -> None:
    app = create_app(gateway_config)
    with respx.mock(assert_all_called=False) as mock:
        mock.post("https://mock.upstream/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(200, json={"choices": [{"message": {"content": "a"}}]}),
                httpx.Response(
                    200,
                    text=(
                        'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
                        "data: [DONE]\n\n"
                    ),
                    headers={"content-type": "text/event-stream"},
                ),
            ]
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer correct-key"},
                json={
                    "model": "openfusion",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )
    await app.state.upstream_client.aclose()
    assert res.status_code == 200


# ---------------------------------------------------------------------------
# /v1/config auth
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_config_no_auth_returns_401(gateway_config: OpenFusionConfig) -> None:
    app = create_app(gateway_config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/v1/config")
    await app.state.upstream_client.aclose()
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_config_wrong_key_returns_401(gateway_config: OpenFusionConfig) -> None:
    app = create_app(gateway_config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/v1/config", headers={"Authorization": "Bearer wrong"})
    await app.state.upstream_client.aclose()
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_config_correct_key_returns_200(gateway_config: OpenFusionConfig) -> None:
    app = create_app(gateway_config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get(
            "/v1/config", headers={"Authorization": "Bearer correct-key"}
        )
    await app.state.upstream_client.aclose()
    assert res.status_code == 200
    assert "panel" in res.json()


# ---------------------------------------------------------------------------
# /v1/estimate auth
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimate_no_auth_returns_401(gateway_config: OpenFusionConfig) -> None:
    app = create_app(gateway_config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/v1/estimate", json={"messages": [{"role": "user", "content": "hi"}]}
        )
    await app.state.upstream_client.aclose()
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_estimate_wrong_key_returns_401(gateway_config: OpenFusionConfig) -> None:
    app = create_app(gateway_config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/v1/estimate",
            headers={"Authorization": "Bearer bad"},
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    await app.state.upstream_client.aclose()
    assert res.status_code == 401


# ---------------------------------------------------------------------------
# /v1/runtime/api-key auth
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_key_endpoint_no_auth_returns_401(
    gateway_config: OpenFusionConfig,
) -> None:
    cfg = _base_config(
        gateway=GatewayAuthConfig(api_keys=["correct-key"]),
        allow_ui_api_key=True,
    )
    app = create_app(cfg)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/v1/runtime/api-key", json={"api_key": "new"})
    await app.state.upstream_client.aclose()
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_runtime_key_endpoint_correct_key_succeeds(
    gateway_config: OpenFusionConfig,
) -> None:
    cfg = _base_config(
        gateway=GatewayAuthConfig(api_keys=["correct-key"]),
        allow_ui_api_key=True,
    )
    app = create_app(cfg)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/v1/runtime/api-key",
            headers={"Authorization": "Bearer correct-key"},
            json={"api_key": "new-key"},
        )
    await app.state.upstream_client.aclose()
    assert res.status_code == 200
    assert res.json()["api_key_set"] is True


@pytest.mark.asyncio
async def test_runtime_api_key_is_scoped_per_client() -> None:
    """One caller's UI-set key must not be visible to, or usable by, another.

    Regression test: app.state.runtime_api_key used to be a single process-wide
    value shared by every caller, so setting a key as one gateway client leaked
    it (and its billing) to every other client's requests.
    """
    cfg = _base_config(
        gateway=GatewayAuthConfig(api_keys=["client-a-key", "client-b-key"]),
        allow_ui_api_key=True,
    )
    app = create_app(cfg)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        set_res = await client.post(
            "/v1/runtime/api-key",
            headers={"Authorization": "Bearer client-a-key"},
            json={"api_key": "sk-belongs-to-a"},
        )
        assert set_res.status_code == 200

        cfg_a = (
            await client.get(
                "/v1/config", headers={"Authorization": "Bearer client-a-key"}
            )
        ).json()
        cfg_b = (
            await client.get(
                "/v1/config", headers={"Authorization": "Bearer client-b-key"}
            )
        ).json()
    await app.state.upstream_client.aclose()

    assert cfg_a["api_key_set"] is True
    assert cfg_b["api_key_set"] is False


@pytest.mark.asyncio
async def test_runtime_api_keys_store_is_bounded() -> None:
    """app.state.runtime_api_keys must not grow without bound.

    With no gateway allowlist configured, /v1/runtime/api-key is reachable
    without authorization, and _client_key() uses the caller-supplied
    Authorization header verbatim as the dict key. Before this was bounded, a
    client sending many distinct Authorization headers could grow this dict
    forever — an unbounded-memory DoS in the (default) zero-config mode.
    """
    from openfusion.server import RUNTIME_API_KEYS_MAX

    cfg = _base_config(allow_ui_api_key=True)
    app = create_app(cfg)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for i in range(RUNTIME_API_KEYS_MAX + 50):
            res = await client.post(
                "/v1/runtime/api-key",
                headers={"Authorization": f"Bearer client-{i}"},
                json={"api_key": f"sk-{i}"},
            )
            assert res.status_code == 200
    await app.state.upstream_client.aclose()

    assert len(app.state.runtime_api_keys) == RUNTIME_API_KEYS_MAX
    # Oldest clients were evicted first (LRU); the most recent survive.
    assert "client-0" not in app.state.runtime_api_keys
    assert f"client-{RUNTIME_API_KEYS_MAX + 49}" in app.state.runtime_api_keys


# ---------------------------------------------------------------------------
# No gateway configured → all endpoints are open
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_gateway_chat_completions_is_open() -> None:
    app = create_app(_base_config())
    with respx.mock(assert_all_called=False) as mock:
        mock.post("https://mock.upstream/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(200, json={"choices": [{"message": {"content": "a"}}]}),
                httpx.Response(
                    200,
                    text=(
                        'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
                        "data: [DONE]\n\n"
                    ),
                    headers={"content-type": "text/event-stream"},
                ),
            ]
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "openfusion",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )
    await app.state.upstream_client.aclose()
    assert res.status_code == 200


# ---------------------------------------------------------------------------
# Concurrency / rate limits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrency_limit_returns_503() -> None:
    """When max_in_flight=1 is exhausted a new request gets 503."""
    cfg = _base_config(limits=LimitsConfig(max_in_flight=1))
    app = create_app(cfg)

    # Directly hold the single slot so the HTTP request sees it full.
    app.state.limiter._in_flight = 1

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "model": "pass-m"},
        )
    await app.state.upstream_client.aclose()
    assert res.status_code == 503
