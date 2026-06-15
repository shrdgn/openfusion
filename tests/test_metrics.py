"""Tests for the metrics registry and the /metrics endpoint."""

from __future__ import annotations

import json

import httpx
import pytest

from openfusion.metrics import METRICS, Metrics


@pytest.fixture(autouse=True)
def _reset_metrics() -> None:
    METRICS.reset()
    yield
    METRICS.reset()


def test_counters_accumulate_by_labels() -> None:
    m = Metrics()
    m.record_request(route="fusion", outcome="success", latency_ms=100)
    m.record_request(route="fusion", outcome="success", latency_ms=300)
    m.record_request(route="pass_through", outcome="error", latency_ms=50)

    assert m.value("openfusion_requests_total", route="fusion", outcome="success") == 2
    assert m.value("openfusion_requests_total", route="pass_through", outcome="error") == 1
    assert m.snapshot()["request_latency_ms"]["fusion"] == {"count": 2, "sum_ms": 400}


def test_upstream_records_tokens_and_cost() -> None:
    m = Metrics()
    m.record_upstream(
        phase="panel",
        outcome="success",
        latency_ms=200,
        usage={"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.002},
    )
    m.record_upstream(
        phase="judge",
        outcome="success",
        latency_ms=400,
        usage={"prompt_tokens": 20, "completion_tokens": 8, "cost": 0.004},
    )

    assert m.value("openfusion_tokens_total", phase="panel", kind="prompt") == 10
    assert m.value("openfusion_tokens_total", phase="judge", kind="completion") == 8
    assert m.value("openfusion_cost_usd_total", phase="panel") == pytest.approx(0.002)
    assert m.value("openfusion_cost_usd_total", phase="judge") == pytest.approx(0.004)


def test_prometheus_render_is_well_formed() -> None:
    m = Metrics()
    m.record_request(route="fusion", outcome="success", latency_ms=120)
    m.record_upstream(
        phase="panel", outcome="success", latency_ms=80, usage={"prompt_tokens": 3}
    )

    text = m.render_prometheus()
    assert "# TYPE openfusion_requests_total counter" in text
    assert 'openfusion_requests_total{outcome="success",route="fusion"} 1' in text
    assert "# TYPE openfusion_request_latency_ms summary" in text
    assert 'openfusion_request_latency_ms_count{route="fusion"} 1' in text
    assert text.endswith("\n")


async def test_metrics_endpoint_exposes_prometheus(client: httpx.AsyncClient) -> None:
    METRICS.record_request(route="fusion", outcome="success", latency_ms=10)
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "openfusion_requests_total" in response.text


async def test_fusion_request_populates_metrics(client: httpx.AsyncClient, mock_router) -> None:
    def upstream_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if payload.get("stream"):
            chunks = (
                'data: {"choices":[{"delta":{"content":"fused"},"finish_reason":null}]}\n\n'
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":4,"completion_tokens":2,'
                '"total_tokens":6,"cost":0.004}}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, text=chunks, headers={"content-type": "text/event-stream"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "panel"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "cost": 0.001,
                },
            },
        )

    mock_router.post("https://mock.upstream/v1/chat/completions").mock(side_effect=upstream_handler)

    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "openfusion",
            "messages": [{"role": "user", "content": "q"}],
            "stream": False,
        },
    )
    assert response.status_code == 200

    assert METRICS.value("openfusion_requests_total", route="fusion", outcome="success") == 1
    # n=3 self-fusion panel members all succeed.
    assert METRICS.value("openfusion_panel_members_total", outcome="success") == 3
    upstream = "openfusion_upstream_requests_total"
    assert METRICS.value(upstream, phase="panel", outcome="success") == 3
    assert METRICS.value(upstream, phase="judge", outcome="success") == 1
    # Panel cost (3 x 0.001) plus judge cost (0.004) are recorded per phase.
    assert METRICS.value("openfusion_cost_usd_total", phase="panel") == pytest.approx(0.003)
    assert METRICS.value("openfusion_cost_usd_total", phase="judge") == pytest.approx(0.004)
