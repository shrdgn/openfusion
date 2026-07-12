"""Error envelope tests."""

from __future__ import annotations

import httpx
import pytest

from openfusion.config import OpenFusionConfig, PanelMember
from openfusion.server import create_app


async def test_missing_model_returns_openai_error(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert "model" in payload["error"]["message"]


def _authed_app(test_config, keys: list[str]):
    test_config.gateway.api_keys = keys
    return create_app(test_config)


async def test_gateway_auth_rejects_invalid_key(test_config, app) -> None:
    authed_app = _authed_app(test_config, ["allowed-key"])
    transport = httpx.ASGITransport(app=authed_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer wrong"},
            json={
                "model": "pass-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


@pytest.mark.asyncio
async def test_gateway_auth_accepts_valid_key(test_config, mock_router) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )
    )
    authed_app = _authed_app(test_config, ["good-key", "other-key"])
    transport = httpx.ASGITransport(app=authed_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer good-key"},
            json={"model": "pass-model", "messages": [{"role": "user", "content": "hi"}]},
        )
    await authed_app.state.upstream_client.aclose()
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_gateway_auth_rejects_key_prefix(test_config, mock_router) -> None:
    """A key that is a strict prefix of a valid key must be rejected."""
    authed_app = _authed_app(test_config, ["long-secret-key"])
    transport = httpx.ASGITransport(app=authed_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer long-secret"},
            json={"model": "pass-model", "messages": [{"role": "user", "content": "hi"}]},
        )
    await authed_app.state.upstream_client.aclose()
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_models_endpoint_requires_gateway_auth(test_config) -> None:
    """/v1/models must be gated when gateway.api_keys is configured."""
    authed_app = _authed_app(test_config, ["secret"])
    transport = httpx.ASGITransport(app=authed_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # No auth header → 401
        unauthenticated = await client.get("/v1/models")
        # Correct key → 200 with model list
        authenticated = await client.get(
            "/v1/models", headers={"Authorization": "Bearer secret"}
        )
    await authed_app.state.upstream_client.aclose()

    assert unauthenticated.status_code == 401
    assert unauthenticated.json()["error"]["code"] == "invalid_api_key"
    assert authenticated.status_code == 200
    assert "data" in authenticated.json()


async def test_malformed_json_body_returns_invalid_request_error(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post(
        "/v1/chat/completions",
        content=b"{not valid json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert "Invalid JSON" in payload["error"]["message"]


async def test_non_object_body_returns_invalid_request_error(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post("/v1/chat/completions", json=["not", "an", "object"])

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert "JSON object" in payload["error"]["message"]


async def test_non_object_override_returns_invalid_request_error(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "pass-model",
            "messages": [{"role": "user", "content": "hi"}],
            "openfusion": "not-an-object",
        },
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert "openfusion override must be an object" in payload["error"]["message"]


async def test_judge_aggregation_without_judge_returns_invalid_request_error(
    mock_router,
) -> None:
    config = OpenFusionConfig(
        panel=[
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m1"),
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m2"),
        ],
        judge=None,
    )
    app = create_app(config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        response = await http_client.post(
            "/v1/chat/completions",
            json={"model": "openfusion", "messages": [{"role": "user", "content": "hi"}]},
        )
    await app.state.upstream_client.aclose()

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert "Judge must be configured" in payload["error"]["message"]


async def test_unexpected_exception_returns_upstream_error(
    client: httpx.AsyncClient, monkeypatch
) -> None:
    """A non-OpenFusionError raised mid-request still yields an OpenAI-shaped error."""

    async def _boom(*_args, **_kwargs):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr("openfusion.server.buffer_synthesis", _boom)

    response = await client.post(
        "/v1/chat/completions",
        json={"model": "openfusion", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 502
    payload = response.json()
    assert payload["error"]["type"] == "upstream_error"
    assert "unexpected failure" in payload["error"]["message"]
