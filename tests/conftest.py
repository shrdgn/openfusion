"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx
import pytest
import respx
from fastapi import FastAPI

from openfusion.config import (
    JudgeConfig,
    OpenFusionConfig,
    PanelMember,
    PassThroughConfig,
    SelfFusionConfig,
    Strategy,
    TimeoutsConfig,
)
from openfusion.server import create_app

EXAMPLE_CONFIG = Path(__file__).resolve().parents[1] / "openfusion.yaml.example"


@pytest.fixture
def example_config_text() -> str:
    return EXAMPLE_CONFIG.read_text(encoding="utf-8")


@pytest.fixture
def test_config() -> OpenFusionConfig:
    return OpenFusionConfig(
        strategy=Strategy.SELF_FUSION,
        panel=[
            PanelMember(
                base_url="https://mock.upstream/v1",
                api_key="panel-key",
                model="test-model",
                label="panel-a",
            )
        ],
        judge=JudgeConfig(
            base_url="https://mock.upstream/v1",
            api_key="judge-key",
            model="judge-model",
        ),
        self_fusion=SelfFusionConfig(n=3),
        timeouts=TimeoutsConfig(member_seconds=5, judge_seconds=5, total_seconds=15),
        pass_through=PassThroughConfig(
            base_url="https://mock.upstream/v1",
            api_key="pass-key",
            model="pass-model",
        ),
    )


@pytest.fixture
def app(test_config: OpenFusionConfig) -> FastAPI:
    return create_app(test_config)


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client


@pytest.fixture
def mock_router() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
