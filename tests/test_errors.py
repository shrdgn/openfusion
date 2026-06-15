"""Error envelope tests."""

from __future__ import annotations

import httpx


async def test_missing_model_returns_openai_error(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert "model" in payload["error"]["message"]


async def test_gateway_auth_rejects_invalid_key(test_config, app) -> None:
    from openfusion.server import create_app

    test_config.gateway.api_keys = ["allowed-key"]
    authed_app = create_app(test_config)
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
