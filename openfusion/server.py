"""FastAPI server for the openfusion proxy."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from openfusion.config import Aggregator, OpenFusionConfig, PanelMember, load_config
from openfusion.cost import CostPolicy, RequestPhase
from openfusion.errors import (
    AuthenticationError,
    InvalidRequestError,
    OpenFusionError,
    UpstreamError,
)
from openfusion.limits import RequestLimiter
from openfusion.metrics import METRICS
from openfusion.router import RouteDecision, route_async
from openfusion.stream import (
    buffer_ranked,
    buffer_synthesis,
    buffer_vote,
    ranked_and_stream,
    synthesize_and_stream,
    vote_and_stream,
)
from openfusion.tools import tools_are_server_executable
from openfusion.upstream import UpstreamClient

FUSION_MODEL = "openfusion"
LANDING_PAGE_DIR = Path(__file__).resolve().parent / "static" / "landing"


def _requires_pass_through_tools(body: dict[str, Any]) -> bool:
    """True when the request carries tools that fusion can't handle.

    A mid-conversation tool exchange (assistant ``tool_calls`` / ``tool`` results)
    or client-side function tools must pass through to a single model. Tools the
    upstream executes server-side (web search/fetch) are fine to fuse, because the
    panel runs them upstream and returns a final text answer.
    """
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("tool_calls") or message.get("role") == "tool":
                return True
    if body.get("functions") or body.get("function_call"):
        return True
    tools = body.get("tools")
    return bool(tools) and not tools_are_server_executable(tools)


def _client_key(authorization: str | None) -> str:
    """Identity for rate limiting: the gateway token, or 'anonymous'."""
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
        if token:
            return token
    return "anonymous"


def _attach_release(response: Any, limiter: RequestLimiter, acquired: bool) -> Any:
    """Release the concurrency slot after the response (incl. stream) finishes."""
    if acquired and getattr(response, "background", None) is None:
        response.background = BackgroundTask(limiter.release, acquired)
    return response


def _validate_gateway_auth(
    config: OpenFusionConfig,
    authorization: str | None,
) -> None:
    if not config.gateway.api_keys:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise AuthenticationError()
    token = authorization.removeprefix("Bearer ").strip()
    if token not in config.gateway.api_keys:
        raise AuthenticationError()


def _error_response(exc: OpenFusionError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


def _record_request(route: str, outcome: str, started: float) -> None:
    METRICS.record_request(
        route=route,
        outcome=outcome,
        latency_ms=(time.perf_counter() - started) * 1000,
    )


def create_app(config: OpenFusionConfig | None = None) -> FastAPI:
    app_config = config or load_config()
    upstream_client = UpstreamClient()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await upstream_client.aclose()

    app = FastAPI(title="openfusion", version="0.1.0", lifespan=lifespan)
    app.state.config = app_config
    app.state.upstream_client = upstream_client
    app.state.limiter = RequestLimiter(app_config.limits)
    app.mount(
        "/landing",
        StaticFiles(directory=LANDING_PAGE_DIR),
        name="landing",
    )

    def get_config() -> OpenFusionConfig:
        return app.state.config

    def get_client() -> UpstreamClient:
        return app.state.upstream_client

    @app.get("/", include_in_schema=False)
    async def landing_page() -> FileResponse:
        return FileResponse(LANDING_PAGE_DIR / "index.html", media_type="text/html")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(
            METRICS.render_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.get("/v1/models")
    async def list_models(cfg: OpenFusionConfig = Depends(get_config)) -> dict[str, Any]:
        models = [
            {
                "id": cfg.fusion_model_name,
                "object": "model",
                "created": 0,
                "owned_by": "openfusion",
            }
        ]
        pass_through = cfg.resolved_pass_through()
        models.append(
            {
                "id": pass_through.model,
                "object": "model",
                "created": 0,
                "owned_by": "pass-through",
            }
        )
        for member in cfg.panel:
            if member.model != pass_through.model:
                models.append(
                    {
                        "id": member.model,
                        "object": "model",
                        "created": 0,
                        "owned_by": "panel",
                    }
                )
        return {"object": "list", "data": models}

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        cfg: OpenFusionConfig = Depends(get_config),
        client: UpstreamClient = Depends(get_client),
        authorization: str | None = Header(default=None),
    ) -> Any:
        started = time.perf_counter()
        route_label = "fusion"
        limiter: RequestLimiter = app.state.limiter
        acquired = False
        try:
            _validate_gateway_auth(cfg, authorization)
            limiter.check_rate(_client_key(authorization))
            body = await request.json()
            if not isinstance(body, dict):
                raise InvalidRequestError("Request body must be a JSON object")

            model = body.get("model")
            if not isinstance(model, str) or not model:
                raise InvalidRequestError("model is required")

            policy = CostPolicy(cfg.cost_controls)
            policy.validate_max_tokens(body)
            stream = bool(body.get("stream", False))

            wants_fusion = model == cfg.fusion_model_name and not _requires_pass_through_tools(
                body
            )
            if wants_fusion and cfg.router.enabled:
                decision = await route_async(body, cfg.router, client)
                if decision == RouteDecision.SOLO:
                    wants_fusion = False

            acquired = limiter.acquire()

            if not wants_fusion:
                route_label = "pass_through"
                limited_body = policy.apply_token_limit(
                    body,
                    RequestPhase.PASS_THROUGH,
                    reject_over_limit=True,
                )
                response = await _pass_through(
                    limited_body, cfg, client, stream=stream, started=started
                )
                if not stream:
                    _record_request(route_label, "success", started)
                return _attach_release(response, limiter, acquired)

            if cfg.aggregator in (Aggregator.JUDGE, Aggregator.RANKED):
                if cfg.judge is None:
                    raise InvalidRequestError("Judge must be configured for judge aggregation")
                policy.apply_token_limit(body, RequestPhase.JUDGE, reject_over_limit=True)

            if stream:
                response = await _fusion_stream(request, body, cfg, client, started=started)
                return _attach_release(response, limiter, acquired)
            if cfg.aggregator == Aggregator.VOTE:
                payload = await buffer_vote(body, cfg, client)
            elif cfg.aggregator == Aggregator.RANKED:
                payload = await buffer_ranked(body, cfg, client)
            else:
                payload = await buffer_synthesis(body, cfg, client)
            _record_request(route_label, "success", started)
            return _attach_release(JSONResponse(content=payload), limiter, acquired)
        except OpenFusionError as exc:
            limiter.release(acquired)
            _record_request(route_label, "error", started)
            return _error_response(exc)
        except json.JSONDecodeError as exc:
            limiter.release(acquired)
            _record_request(route_label, "error", started)
            return _error_response(InvalidRequestError(f"Invalid JSON: {exc}"))
        except Exception as exc:  # noqa: BLE001
            limiter.release(acquired)
            _record_request(route_label, "error", started)
            return _error_response(UpstreamError(str(exc)))

    return app


async def _pass_through(
    body: dict[str, Any],
    config: OpenFusionConfig,
    client: UpstreamClient,
    *,
    stream: bool,
    started: float,
) -> Any:
    pass_through = config.resolved_pass_through()
    member = PanelMember(
        base_url=pass_through.base_url,
        api_key=pass_through.api_key,
        model=body.get("model", pass_through.model),
        label="pass-through",
    )

    if body.get("model") != config.fusion_model_name:
        member = member.model_copy(update={"model": str(body["model"])})

    payload = {**body, "model": member.model}
    result = await client.chat_completion(
        member,
        payload,
        stream=stream,
        phase=RequestPhase.PASS_THROUGH,
    )

    if stream:
        if not hasattr(result, "__aiter__"):
            raise UpstreamError("Expected streaming upstream response")

        async def event_stream() -> AsyncIterator[str]:
            outcome = "success"
            try:
                async for chunk in result:
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception:
                outcome = "error"
                raise
            finally:
                _record_request("pass_through", outcome, started)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    if not isinstance(result, dict):
        raise UpstreamError("Expected JSON upstream response")
    return JSONResponse(content=result)


async def _fusion_stream(
    request: Request,
    body: dict[str, Any],
    config: OpenFusionConfig,
    client: UpstreamClient,
    *,
    started: float,
) -> StreamingResponse:
    cancel_event = asyncio.Event()

    if config.aggregator == Aggregator.VOTE:
        streamer = vote_and_stream
    elif config.aggregator == Aggregator.RANKED:
        streamer = ranked_and_stream
    else:
        streamer = synthesize_and_stream

    async def event_stream() -> AsyncIterator[str]:
        task = asyncio.create_task(_watch_disconnect(request, cancel_event))
        outcome = "success"
        try:
            async for line in streamer(
                body,
                config,
                client,
                cancel_event=cancel_event,
            ):
                if cancel_event.is_set():
                    break
                yield line
        except Exception:
            outcome = "error"
            raise
        finally:
            cancel_event.set()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            _record_request("fusion", outcome, started)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


async def _watch_disconnect(request: Request, cancel_event: asyncio.Event) -> None:
    while not cancel_event.is_set():
        if await request.is_disconnected():
            cancel_event.set()
            return
        await asyncio.sleep(0.2)
