"""FastAPI server for the openfusion proxy."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from openfusion.config import (
    _PRESETS,
    OPENROUTER_BASE_URL,
    Aggregator,
    LimitsConfig,
    OpenFusionConfig,
    PanelMember,
    PassThroughConfig,
    ResponseCacheConfig,
    RouteModel,
    load_config,
)
from openfusion.cost import CostPolicy, RequestPhase
from openfusion.errors import (
    AuthenticationError,
    InvalidRequestError,
    OpenFusionError,
    UpstreamError,
)
from openfusion.estimate import build_estimate
from openfusion.limits import RequestLimiter
from openfusion.metrics import METRICS
from openfusion.overrides import apply_overrides, fill_missing_keys, is_missing_api_key
from openfusion.pricing import get_prices
from openfusion.responsecache import ResponseCache, cache_key
from openfusion.router import RouteDecision, route_request
from openfusion.stream import (
    buffer_ranked,
    buffer_synthesis,
    buffer_vote,
    cached_response_dict,
    capture_stream,
    ranked_and_stream,
    replay_cached_stream,
    synthesize_and_stream,
    vote_and_stream,
)
from openfusion.tools import tools_are_server_executable
from openfusion.upstream import UpstreamClient

FUSION_MODEL = "openfusion"
STATIC_DIR = Path(__file__).resolve().parent / "static"
PLAYGROUND_DIR = STATIC_DIR / "playground"


def _preset_summary() -> dict[str, Any]:
    return {
        name.value: {
            "panel": spec["panel_models"],
            "judge": spec["judge_model"],
        }
        for name, spec in _PRESETS.items()
    }


def _active_config_payload(cfg: OpenFusionConfig, runtime_key: str | None) -> dict[str, Any]:
    needs_key = is_missing_api_key(cfg) and not runtime_key
    return {
        "preset": cfg.preset.value if cfg.preset else None,
        "strategy": cfg.strategy.value,
        "aggregator": cfg.aggregator.value,
        "panel": [member.model for member in cfg.panel],
        "judge": cfg.judge.model if cfg.judge else None,
        "tools": {"web_search": cfg.tools.web_search, "web_fetch": cfg.tools.web_fetch},
        "allow_request_overrides": cfg.allow_request_overrides,
        "allow_ui_api_key": cfg.allow_ui_api_key,
        "needs_api_key": needs_key,
        "api_key_set": bool(runtime_key),
        "presets": _preset_summary(),
        "fusion_model": cfg.fusion_model_name,
    }


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
    # Compare against every key without short-circuiting so the response time
    # doesn't reveal which positions in the list are occupied by valid keys.
    token_b = token.encode()
    valid = False
    for key in config.gateway.api_keys:
        if hmac.compare_digest(token_b, key.encode()):
            valid = True
    if not valid:
        raise AuthenticationError()


def _error_response(exc: OpenFusionError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


def _record_request(route: str, outcome: str, started: float) -> None:
    METRICS.record_request(
        route=route,
        outcome=outcome,
        latency_ms=(time.perf_counter() - started) * 1000,
    )


# A host app can supply a per-request config (e.g. the authenticated user's key
# and recipe) instead of one app-wide config — see docs/EMBEDDING.md.
ConfigResolver = Callable[[Request], Awaitable[OpenFusionConfig]]
# Called after a fused request with its usage/cost, for metering.
UsageCallback = Callable[[Request, "dict[str, Any] | None"], Awaitable[None]]


def create_app(
    config: OpenFusionConfig | None = None,
    *,
    config_resolver: ConfigResolver | None = None,
    usage_callback: UsageCallback | None = None,
) -> FastAPI:
    # With a resolver, a static config is optional (the resolver provides one per
    # request); otherwise fall back to loading from disk/zero-config.
    app_config = config
    if app_config is None and config_resolver is None:
        app_config = load_config()
    upstream_client = UpstreamClient()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await upstream_client.aclose()

    app = FastAPI(title="openfusion", version="0.1.0", lifespan=lifespan)
    app.state.config = app_config
    app.state.config_resolver = config_resolver
    app.state.usage_callback = usage_callback
    app.state.upstream_client = upstream_client
    app.state.limiter = RequestLimiter(app_config.limits if app_config else LimitsConfig())
    rc = app_config.response_cache if app_config else ResponseCacheConfig()
    app.state.response_cache = ResponseCache(rc.ttl_seconds, rc.max_entries)
    app.state.runtime_api_key = None
    app.mount(
        "/playground",
        StaticFiles(directory=PLAYGROUND_DIR, html=True),
        name="playground",
    )

    @app.exception_handler(OpenFusionError)
    async def _openfusion_error_handler(_: Request, exc: OpenFusionError) -> JSONResponse:
        return _error_response(exc)

    async def get_config(request: Request) -> OpenFusionConfig:
        resolver: ConfigResolver | None = app.state.config_resolver
        if resolver is not None:
            return await resolver(request)
        if app.state.config is None:
            raise UpstreamError("No configuration available")
        return app.state.config

    def get_client() -> UpstreamClient:
        return app.state.upstream_client

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/playground/")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(
            METRICS.render_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.get("/v1/config")
    async def active_config(cfg: OpenFusionConfig = Depends(get_config)) -> dict[str, Any]:
        return _active_config_payload(cfg, app.state.runtime_api_key)

    @app.post("/v1/estimate")
    async def estimate(
        request: Request,
        cfg: OpenFusionConfig = Depends(get_config),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _validate_gateway_auth(cfg, authorization)
        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise InvalidRequestError(f"Invalid JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise InvalidRequestError("Request body must be a JSON object")
        override = body.pop("openfusion", None)
        if isinstance(override, dict) and cfg.allow_request_overrides:
            cfg = apply_overrides(cfg, override)
        base = cfg.panel[0].base_url if cfg.panel else OPENROUTER_BASE_URL
        prices = await get_prices(base)
        return build_estimate(body, cfg, prices)

    @app.post("/v1/runtime/api-key")
    async def set_runtime_api_key(
        request: Request,
        cfg: OpenFusionConfig = Depends(get_config),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _validate_gateway_auth(cfg, authorization)
        if not cfg.allow_ui_api_key:
            raise OpenFusionError(
                "Setting the API key from the UI is disabled on this server",
                error_type="permission_error",
                code="ui_key_disabled",
                status_code=403,
            )
        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise InvalidRequestError(f"Invalid JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise InvalidRequestError("Request body must be a JSON object")
        key = str(body.get("api_key", "")).strip()
        app.state.runtime_api_key = key or None
        return {"ok": True, "api_key_set": bool(key)}

    @app.get("/v1/models")
    async def list_models(
        cfg: OpenFusionConfig = Depends(get_config),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _validate_gateway_auth(cfg, authorization)
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

            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise InvalidRequestError("messages is required and must be a non-empty array")

            override = body.pop("openfusion", None)
            if override is not None:
                if not isinstance(override, dict):
                    raise InvalidRequestError("openfusion override must be an object")
                if not cfg.allow_request_overrides:
                    raise InvalidRequestError(
                        "Per-request overrides are disabled on this server"
                    )
                cfg = apply_overrides(cfg, override)

            cfg = fill_missing_keys(cfg, app.state.runtime_api_key)
            if is_missing_api_key(cfg):
                raise InvalidRequestError(
                    "No upstream API key configured. Add one in the playground "
                    "or set OPENROUTER_API_KEY.",
                    code="no_api_key",
                )

            policy = CostPolicy(cfg.cost_controls)
            policy.validate_max_tokens(body)
            stream = bool(body.get("stream", False))

            wants_fusion = model == cfg.fusion_model_name and not _requires_pass_through_tools(
                body
            )
            routed: PassThroughConfig | None = None
            if wants_fusion and cfg.router.enabled:
                decision, route_model = await route_request(body, cfg.router, client)
                if decision == RouteDecision.SOLO:
                    wants_fusion = False
                    routed = _routed_pass_through(cfg, route_model)

            acquired = limiter.acquire()

            if not wants_fusion:
                route_label = "pass_through"
                limited_body = policy.apply_token_limit(
                    body,
                    RequestPhase.PASS_THROUGH,
                    reject_over_limit=True,
                )
                response = await _pass_through(
                    limited_body, cfg, client, stream=stream, started=started, override=routed
                )
                if not stream:
                    _record_request(route_label, "success", started)
                return _attach_release(response, limiter, acquired)

            if cfg.aggregator in (Aggregator.JUDGE, Aggregator.RANKED):
                if cfg.judge is None:
                    raise InvalidRequestError("Judge must be configured for judge aggregation")
                policy.apply_token_limit(body, RequestPhase.JUDGE, reject_over_limit=True)

            cache = app.state.response_cache
            cache_k = cache_key(body, cfg) if cfg.response_cache.enabled else None

            async def fire_usage(usage: dict[str, Any] | None) -> None:
                cb = app.state.usage_callback
                if cb is not None:
                    with contextlib.suppress(Exception):
                        await cb(request, usage)

            async def on_complete(content: str, usage: dict[str, Any] | None) -> None:
                if cache_k is not None:
                    cache.put(cache_k, {"content": content, "usage": usage})
                await fire_usage(usage)

            if cache_k is not None and (hit := cache.get(cache_k)) is not None:
                _record_request(route_label, "success", started)
                await fire_usage(hit.get("usage"))
                model_name = cfg.fusion_model_name
                if stream:
                    cached = StreamingResponse(
                        replay_cached_stream(hit["content"], hit.get("usage"), model_name),
                        media_type="text/event-stream",
                    )
                    return _attach_release(cached, limiter, acquired)
                payload = cached_response_dict(hit["content"], hit.get("usage"), model_name)
                return _attach_release(JSONResponse(content=payload), limiter, acquired)

            if stream:
                response = await _fusion_stream(
                    request, body, cfg, client, started=started, on_complete=on_complete
                )
                return _attach_release(response, limiter, acquired)
            if cfg.aggregator == Aggregator.VOTE:
                payload = await buffer_vote(body, cfg, client)
            elif cfg.aggregator == Aggregator.RANKED:
                payload = await buffer_ranked(body, cfg, client)
            else:
                payload = await buffer_synthesis(body, cfg, client)
            await on_complete(
                (payload.get("choices") or [{}])[0].get("message", {}).get("content") or "",
                payload.get("usage"),
            )
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


def _routed_pass_through(
    config: OpenFusionConfig, route_model: RouteModel | None
) -> PassThroughConfig | None:
    """Build a pass-through target for a router-selected model (or None for default)."""
    if route_model is None:
        return None
    base = config.resolved_pass_through()
    return PassThroughConfig(
        base_url=route_model.base_url or base.base_url,
        api_key=route_model.api_key or base.api_key,
        model=route_model.model,
    )


async def _pass_through(
    body: dict[str, Any],
    config: OpenFusionConfig,
    client: UpstreamClient,
    *,
    stream: bool,
    started: float,
    override: PassThroughConfig | None = None,
) -> Any:
    pass_through = override or config.resolved_pass_through()
    requested = body.get("model")
    if override is not None:
        chosen = override.model
    elif isinstance(requested, str) and requested != config.fusion_model_name:
        chosen = requested  # a direct pass-through of a named model
    else:
        chosen = pass_through.model  # router/solo: use the configured single model
    member = PanelMember(
        base_url=pass_through.base_url,
        api_key=pass_through.api_key,
        model=chosen,
        label="pass-through",
    )

    payload = {**body, "model": chosen}
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
    on_complete: Callable[[str, dict[str, Any] | None], Awaitable[None]] | None = None,
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

    lines = event_stream()
    if on_complete is not None:
        lines = capture_stream(lines, on_complete)
    return StreamingResponse(lines, media_type="text/event-stream")


async def _watch_disconnect(request: Request, cancel_event: asyncio.Event) -> None:
    while not cancel_event.is_set():
        if await request.is_disconnected():
            cancel_event.set()
            return
        await asyncio.sleep(0.2)
