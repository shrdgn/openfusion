"""CLI entrypoint for openfusion."""

from __future__ import annotations

import argparse
import os

import uvicorn

from openfusion.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the openfusion proxy server")
    parser.add_argument("--host", default=os.environ.get("OPENFUSION_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("OPENFUSION_PORT", "8000")))
    parser.add_argument("--config", default=os.environ.get("OPENFUSION_CONFIG"))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    load_config(args.config)
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
