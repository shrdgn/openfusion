"""CLI entrypoint for openfusion."""

from __future__ import annotations

import argparse
import os
import sys

import uvicorn
from pydantic import ValidationError

from openfusion.config import OpenFusionConfig, load_config


def _summarize_config(config: OpenFusionConfig, host: str, port: int) -> str:
    """A short, human-readable summary of what's about to run."""
    judge = config.judge.model if config.judge else "—"
    panel = ", ".join(member.model for member in config.panel) or "—"
    recipe = (
        f"preset={config.preset.value}"
        if config.preset
        else f"strategy={config.strategy.value}"
    )
    tools = "web search+fetch" if config.tools.web_search else "off"
    lines = [
        "openfusion is starting",
        f"  recipe:   {recipe}  (aggregator={config.aggregator.value})",
        f"  panel:    {panel}",
        f"  judge:    {judge}",
        f"  tools:    {tools}",
        f"  listening on http://{host}:{port}",
        f'  call it with model="{config.fusion_model_name}" against http://{host}:{port}/v1',
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the openfusion proxy server")
    parser.add_argument("--host", default=os.environ.get("OPENFUSION_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("OPENFUSION_PORT", "8000")))
    parser.add_argument("--config", default=os.environ.get("OPENFUSION_CONFIG"))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    # Ensure the uvicorn factory (which re-loads config in the worker) honors
    # an explicit --config, not just the OPENFUSION_CONFIG env var.
    if args.config:
        os.environ["OPENFUSION_CONFIG"] = args.config

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        print(f"openfusion: could not load configuration.\n{exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(_summarize_config(config, args.host, args.port), file=sys.stderr)
    os.environ["OPENFUSION_LOADED_CONFIG"] = "1"

    uvicorn.run(
        "openfusion.server:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=os.environ.get("OPENFUSION_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
