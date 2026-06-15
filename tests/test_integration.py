"""End-to-end integration tests."""

from __future__ import annotations

import httpx


async def test_healthz(client: httpx.AsyncClient) -> None:
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_models_list_includes_openfusion(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/models")
    assert response.status_code == 200
    ids = {item["id"] for item in response.json()["data"]}
    assert "openfusion" in ids
