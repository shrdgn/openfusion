"""Persist the OpenRouter API key locally so the CLI doesn't re-prompt each run.

Stored as a single line in ``$XDG_CONFIG_HOME/openfusion/credentials`` (default
``~/.config/openfusion/credentials``) with ``600`` permissions. This is a
convenience for local/single-user use; a shared server should use env vars or a
config file instead.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path


def credentials_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base).expanduser() / "openfusion" / "credentials"


def load_saved_key() -> str | None:
    try:
        key = credentials_path().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return key or None


def save_key(key: str) -> None:
    key = key.strip()
    if not key:
        return
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(key + "\n", encoding="utf-8")
    with contextlib.suppress(OSError):
        path.chmod(0o600)


def clear_key() -> None:
    with contextlib.suppress(FileNotFoundError):
        credentials_path().unlink()
