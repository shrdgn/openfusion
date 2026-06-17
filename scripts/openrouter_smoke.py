#!/usr/bin/env python3
"""Opt-in live OpenRouter smoke test for openfusion."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

import httpx

from openfusion.config import load_config
from openfusion.server import create_app


def _require_credit_opt_in(args: argparse.Namespace) -> None:
    if not args.yes_spend_credits:
        raise SystemExit("Refusing to spend credits without --yes-spend-credits.")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is required for live OpenRouter smoke tests.")


async def _post_chat(
    client: httpx.AsyncClient,
    payload: dict[str, Any],
    *,
    stream: bool,
) -> dict[str, Any]:
    response = await client.post("/v1/chat/completions", json=payload, timeout=180.0)
    response.raise_for_status()
    if not stream:
        return response.json()

    content_parts: list[str] = []
    progress: list[dict[str, Any]] = []
    usage: list[dict[str, Any]] = []
    current_event: str | None = None
    async for line in response.aiter_lines():
        if not line:
            continue
        if line.startswith("event: "):
            current_event = line[7:]
            continue
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if "error" in chunk:
            raise RuntimeError(chunk["error"]["message"])
        if current_event == "progress":
            progress.append(chunk)
            current_event = None
            continue
        if current_event == "usage":
            usage.append(chunk)
            current_event = None
            continue
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        if delta.get("content"):
            content_parts.append(str(delta["content"]))
    return {"content": "".join(content_parts), "progress": progress, "usage": usage}


def _assert_contains(actual: str, expected: str) -> None:
    if expected not in actual:
        raise AssertionError(f"Expected {expected!r} in response, got {actual!r}")


async def run(args: argparse.Namespace) -> None:
    _require_credit_opt_in(args)
    config = load_config(args.config)
    app = create_app(config)
    transport = httpx.ASGITransport(app=app)
    headers = {"Authorization": f"Bearer {args.gateway_key}"}

    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://openfusion.local",
            headers=headers,
        ) as client:
            health = await client.get("/healthz")
            health.raise_for_status()
            print(f"healthz={health.json()['status']}")

            pass_model = config.resolved_pass_through().model
            pass_payload = {
                "model": pass_model,
                "messages": [
                    {"role": "user", "content": "Reply with exactly: openrouter-smoke-ok"}
                ],
                "temperature": 0,
                "max_tokens": args.pass_max_tokens,
                "stream": False,
            }
            pass_result = await _post_chat(client, pass_payload, stream=False)
            pass_content = pass_result["choices"][0]["message"]["content"]
            _assert_contains(pass_content, "openrouter-smoke-ok")
            print(f"pass_through_model={pass_result.get('model')}")
            print(f"pass_through_content={pass_content}")
            print(f"pass_through_usage={pass_result.get('usage')}")

            fusion_payload = {
                "model": config.fusion_model_name,
                "messages": [{"role": "user", "content": "Reply with exactly: fusion-smoke-ok"}],
                "temperature": 0,
                "max_tokens": args.fusion_max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            fusion_result = await _post_chat(client, fusion_payload, stream=True)
            fusion_content = fusion_result["content"]
            _assert_contains(fusion_content, "fusion-smoke-ok")
            print(f"fusion_progress={fusion_result['progress']}")
            print(f"fusion_content={fusion_content}")
            print(f"fusion_usage={fusion_result['usage']}")
    finally:
        await app.state.upstream_client.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a live, low-token OpenRouter smoke test.")
    parser.add_argument("--config", default="examples/dev.yaml.example")
    parser.add_argument("--gateway-key", default="smoke-key")
    parser.add_argument("--pass-max-tokens", type=int, default=16)
    parser.add_argument("--fusion-max-tokens", type=int, default=20)
    parser.add_argument("--yes-spend-credits", action="store_true")
    args = parser.parse_args()

    try:
        asyncio.run(run(args))
    except Exception as exc:  # noqa: BLE001
        print(f"openrouter_smoke_failed={exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
