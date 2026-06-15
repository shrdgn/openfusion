"""FastAPI server for the openfusion proxy."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from openfusion.config import OpenFusionConfig, PanelMember, load_config
from openfusion.errors import (
    AuthenticationError,
    InvalidRequestError,
    OpenFusionError,
    UpstreamError,
)
from openfusion.stream import buffer_synthesis, synthesize_and_stream
from openfusion.upstream import UpstreamClient

FUSION_MODEL = "openfusion"


def _has_tool_calls(body: dict[str, Any]) -> bool:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("tool_calls"):
            return True
        if message.get("role") == "tool":
            return True
    return bool(body.get("tools") or body.get("functions"))


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

    def get_config() -> OpenFusionConfig:
        return app.state.config

    def get_client() -> UpstreamClient:
        return app.state.upstream_client

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

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
        try:
            _validate_gateway_auth(cfg, authorization)
            body = await request.json()
            if not isinstance(body, dict):
                raise InvalidRequestError("Request body must be a JSON object")

            model = body.get("model")
            if not isinstance(model, str) or not model:
                raise InvalidRequestError("model is required")

            stream = bool(body.get("stream", False))

            if model != cfg.fusion_model_name or _has_tool_calls(body):
                return await _pass_through(body, cfg, client, stream=stream)

            if cfg.judge is None:
                raise InvalidRequestError("Judge must be configured for fusion requests")

            if stream:
                return await _fusion_stream(request, body, cfg, client)
            payload = await buffer_synthesis(body, cfg, client)
            return JSONResponse(content=payload)
        except OpenFusionError as exc:
            return _error_response(exc)
        except json.JSONDecodeError as exc:
            return _error_response(InvalidRequestError(f"Invalid JSON: {exc}"))
        except Exception as exc:  # noqa: BLE001
            return _error_response(UpstreamError(str(exc)))

    return app


async def _pass_through(
    body: dict[str, Any],
    config: OpenFusionConfig,
    client: UpstreamClient,
    *,
    stream: bool,
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
    result = await client.chat_completion(member, payload, stream=stream)

    if stream:
        if not hasattr(result, "__aiter__"):
            raise UpstreamError("Expected streaming upstream response")

        async def event_stream() -> AsyncIterator[str]:
            async for chunk in result:
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    if not isinstance(result, dict):
        raise UpstreamError("Expected JSON upstream response")
    return JSONResponse(content=result)


async def _fusion_stream(
    request: Request,
    body: dict[str, Any],
    config: OpenFusionConfig,
    client: UpstreamClient,
) -> StreamingResponse:
    cancel_event = asyncio.Event()

    async def event_stream() -> AsyncIterator[str]:
        task = asyncio.create_task(_watch_disconnect(request, cancel_event))
        try:
            async for line in synthesize_and_stream(
                body,
                config,
                client,
                cancel_event=cancel_event,
            ):
                if cancel_event.is_set():
                    break
                yield line
        finally:
            cancel_event.set()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    return StreamingResponse(event_stream(), media_type="text/event-stream")


async def _watch_disconnect(request: Request, cancel_event: asyncio.Event) -> None:
    while not cancel_event.is_set():
        if await request.is_disconnected():
            cancel_event.set()
            return
        await asyncio.sleep(0.2)
