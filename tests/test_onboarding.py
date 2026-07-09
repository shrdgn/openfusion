"""Zero-config quick start and runtime API-key onboarding."""

from __future__ import annotations

import httpx
import pytest

from openfusion.config import OpenFusionConfig, quickstart_config
from openfusion.overrides import fill_missing_keys, is_missing_api_key
from openfusion.server import create_app


def test_quickstart_config_without_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = quickstart_config()
    assert cfg.preset is not None
    assert len(cfg.panel) == 3
    assert cfg.allow_ui_api_key is True
    assert cfg.allow_request_overrides is True
    assert is_missing_api_key(cfg) is True  # no key yet


def test_quickstart_config_uses_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    cfg = quickstart_config()
    assert is_missing_api_key(cfg) is False
    assert all(m.api_key == "sk-test" for m in cfg.panel)


def test_fill_missing_keys() -> None:
    cfg = quickstart_config()
    for member in cfg.panel:
        member.api_key = ""
    if cfg.judge:
        cfg.judge.api_key = ""
    if cfg.pass_through:
        cfg.pass_through.api_key = ""
    filled = fill_missing_keys(cfg, "sk-runtime")
    assert is_missing_api_key(filled) is False
    assert filled.judge is not None and filled.judge.api_key == "sk-runtime"


def _quickstart_app() -> object:
    cfg = quickstart_config()
    # ensure keys are empty regardless of ambient env
    for member in cfg.panel:
        member.api_key = ""
    if cfg.judge:
        cfg.judge.api_key = ""
    if cfg.pass_through:
        cfg.pass_through.api_key = ""
    return create_app(cfg)


async def test_v1_config_reports_needs_api_key() -> None:
    app = _quickstart_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        body = (await http_client.get("/v1/config")).json()
    await app.state.upstream_client.aclose()
    assert body["needs_api_key"] is True
    assert body["allow_ui_api_key"] is True


async def test_runtime_api_key_clears_needs_flag() -> None:
    app = _quickstart_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        set_res = await http_client.post("/v1/runtime/api-key", json={"api_key": "sk-live"})
        assert set_res.status_code == 200
        assert set_res.json()["api_key_set"] is True
        cfg = (await http_client.get("/v1/config")).json()
    await app.state.upstream_client.aclose()
    assert cfg["needs_api_key"] is False
    assert cfg["api_key_set"] is True


async def test_runtime_api_key_rejected_when_disabled() -> None:
    cfg = OpenFusionConfig.model_validate(
        {
            "panel": [{"base_url": "https://x/v1", "api_key": "k", "model": "m"}],
            "allow_ui_api_key": False,
        }
    )
    app = create_app(cfg)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        res = await http_client.post("/v1/runtime/api-key", json={"api_key": "x"})
    await app.state.upstream_client.aclose()
    assert res.status_code == 403
    assert res.json()["error"]["code"] == "ui_key_disabled"


async def test_runtime_api_key_can_be_cleared() -> None:
    """Posting an empty api_key removes any previously-set runtime key."""
    app = _quickstart_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        await http_client.post("/v1/runtime/api-key", json={"api_key": "sk-live"})
        clear_res = await http_client.post("/v1/runtime/api-key", json={"api_key": ""})
        cfg = (await http_client.get("/v1/config")).json()
    await app.state.upstream_client.aclose()
    assert clear_res.status_code == 200
    assert clear_res.json()["api_key_set"] is False
    assert cfg["needs_api_key"] is True


async def test_runtime_api_key_rejects_invalid_json() -> None:
    app = _quickstart_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        res = await http_client.post(
            "/v1/runtime/api-key",
            content=b"{not json",
            headers={"Content-Type": "application/json"},
        )
    await app.state.upstream_client.aclose()
    assert res.status_code == 400
    assert res.json()["error"]["type"] == "invalid_request_error"


async def test_runtime_api_key_rejects_non_object_body() -> None:
    app = _quickstart_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        res = await http_client.post("/v1/runtime/api-key", json=["not", "an", "object"])
    await app.state.upstream_client.aclose()
    assert res.status_code == 400


async def test_request_without_key_returns_clear_error() -> None:
    app = _quickstart_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        res = await http_client.post(
            "/v1/chat/completions",
            json={"model": "openfusion", "messages": [{"role": "user", "content": "hi"}]},
        )
    await app.state.upstream_client.aclose()
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "no_api_key"
