"""Saved-key persistence for the CLI."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from openfusion.config import quickstart_config
from openfusion.credentials import clear_key, credentials_path, load_saved_key, save_key
from openfusion.overrides import is_missing_api_key


@pytest.fixture(autouse=True)
def _isolated_config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))


def test_save_and_load_roundtrip() -> None:
    assert load_saved_key() is None
    save_key("sk-or-123")
    assert load_saved_key() == "sk-or-123"


def test_saved_key_file_is_private() -> None:
    save_key("sk-or-123")
    mode = stat.S_IMODE(credentials_path().stat().st_mode)
    assert mode == 0o600


def test_clear_key() -> None:
    save_key("sk-or-123")
    clear_key()
    assert load_saved_key() is None
    clear_key()  # idempotent


def test_blank_key_is_not_saved() -> None:
    save_key("   ")
    assert load_saved_key() is None


def test_quickstart_uses_saved_key_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    save_key("sk-saved")
    config = quickstart_config()
    assert is_missing_api_key(config) is False
    assert all(member.api_key == "sk-saved" for member in config.panel)


def test_env_key_takes_precedence_over_saved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    save_key("sk-saved")
    config = quickstart_config()
    assert all(member.api_key == "sk-env" for member in config.panel)
