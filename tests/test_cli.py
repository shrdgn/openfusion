"""CLI summary and config-error friendliness tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import pytest

from openfusion import cli, credentials
from openfusion.cli import (
    _ask_cli,
    _chat_turn,
    _models_markup,
    _run_ask,
    _summarize_config,
    build_setup_yaml,
    main,
    run_ask,
    run_chat,
    run_server,
    run_setup,
)
from openfusion.config import (
    Aggregator,
    JudgeConfig,
    OpenFusionConfig,
    PanelMember,
    Strategy,
    load_config,
)


async def test_run_ask_prints_fused_answer(mock_router, capsys: pytest.CaptureFixture[str]) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "the answer"}}]}
        )
    )
    config = OpenFusionConfig(
        strategy=Strategy.PANEL,
        aggregator=Aggregator.VOTE,  # avoid the judge stream for a clean capture
        panel=[
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m1"),
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m2"),
        ],
        judge=JudgeConfig(base_url="https://mock.upstream/v1", api_key="k", model="j"),
    )

    await _run_ask("what is 2+2?", config)

    assert "the answer" in capsys.readouterr().out


async def test_chat_turn_streams_and_returns_answer(mock_router) -> None:
    mock_router.post("https://mock.upstream/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "fused reply"}}]}
        )
    )
    config = OpenFusionConfig(
        strategy=Strategy.PANEL,
        aggregator=Aggregator.VOTE,
        panel=[
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m1"),
            PanelMember(base_url="https://mock.upstream/v1", api_key="k", model="m2"),
        ],
        judge=JudgeConfig(base_url="https://mock.upstream/v1", api_key="k", model="j"),
    )

    answer = await _chat_turn([{"role": "user", "content": "hi"}], config)

    assert "fused reply" in answer


def test_setup_yaml_loads_into_valid_config(tmp_path: Path) -> None:
    config_path = tmp_path / "openfusion.yaml"
    config_path.write_text(build_setup_yaml("budget", "sk-xyz"), encoding="utf-8")

    config = load_config(config_path)

    assert len(config.panel) == 3
    assert all(member.api_key == "sk-xyz" for member in config.panel)
    assert config.judge is not None and config.judge.api_key == "sk-xyz"
    assert config.tools.web_search is True


def test_summarize_config_reports_preset_and_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-key")
    config_path = tmp_path / "openfusion.yaml"
    config_path.write_text("preset: budget\n", encoding="utf-8")
    config = load_config(config_path)

    summary = _summarize_config(config, "0.0.0.0", 8000)

    assert "preset=budget" in summary
    assert "web search+fetch" in summary
    assert 'model="openfusion"' in summary
    assert "http://0.0.0.0:8000" in summary


def test_missing_config_file_has_actionable_hint(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError, match="cp examples/preset.yaml.example"):
        load_config(missing)


def test_missing_env_var_hint_includes_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config_path = tmp_path / "openfusion.yaml"
    config_path.write_text(
        """
panel:
  - base_url: https://example.com/v1
    api_key: ${OPENROUTER_API_KEY}
    model: test
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="export OPENROUTER_API_KEY"):
        load_config(config_path)


# ---------------------------------------------------------------------------
# _models_markup
# ---------------------------------------------------------------------------


def test_models_markup_lists_panel_and_judge() -> None:
    config = OpenFusionConfig(
        panel=[
            PanelMember(base_url="https://example.com/v1", api_key="k", model="m1"),
            PanelMember(base_url="https://example.com/v1", api_key="k", model="m2"),
        ],
        judge=JudgeConfig(base_url="https://example.com/v1", api_key="k", model="j"),
        aggregator=Aggregator.VOTE,
    )
    markup = _models_markup(config)
    assert "m1, m2" in markup
    assert "j" in markup
    assert "vote" in markup


def test_models_markup_handles_no_judge() -> None:
    config = OpenFusionConfig(
        panel=[PanelMember(base_url="https://example.com/v1", api_key="k", model="m1")],
    )
    assert "—" in _models_markup(config)


# ---------------------------------------------------------------------------
# _ask_cli argv parsing
# ---------------------------------------------------------------------------


def test_ask_cli_forwards_prompt_config_and_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        cli,
        "run_ask",
        lambda prompt, config_path, max_tokens: captured.update(
            prompt=prompt, config_path=config_path, max_tokens=max_tokens
        ),
    )

    _ask_cli(["what is 2+2?", "--config", "my.yaml", "--max-tokens", "50"])

    assert captured == {"prompt": "what is 2+2?", "config_path": "my.yaml", "max_tokens": 50}


def test_ask_cli_requires_a_prompt(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _ask_cli([])

    assert excinfo.value.code == 2
    assert "the following arguments are required: prompt" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# run_ask error paths
# ---------------------------------------------------------------------------


def test_run_ask_exits_when_config_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(SystemExit) as excinfo:
        run_ask("hi", str(missing), None)

    assert excinfo.value.code == 1
    assert "could not load configuration" in capsys.readouterr().err


def test_run_ask_exits_when_no_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_saved_key", lambda: None)
    config_path = tmp_path / "openfusion.yaml"
    config_path.write_text(
        'panel:\n  - base_url: https://example.com/v1\n    api_key: ""\n    model: m\n',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        run_ask("hi", str(config_path), None)

    assert excinfo.value.code == 1
    assert "no upstream API key" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# run_server argv parsing and error paths
# ---------------------------------------------------------------------------


def test_run_server_sets_config_env_and_forwards_uvicorn_kwargs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENFUSION_CONFIG", raising=False)
    monkeypatch.delenv("OPENFUSION_LOADED_CONFIG", raising=False)
    config_path = tmp_path / "openfusion.yaml"
    config_path.write_text("preset: budget\n", encoding="utf-8")

    captured: dict = {}
    monkeypatch.setattr(cli.uvicorn, "run", lambda *_args, **kwargs: captured.update(kwargs))

    run_server(["--config", str(config_path), "--port", "9001", "--no-open"])

    assert os.environ["OPENFUSION_CONFIG"] == str(config_path)
    assert os.environ["OPENFUSION_LOADED_CONFIG"] == "1"
    assert captured["port"] == 9001
    assert captured["host"] == "0.0.0.0"


def test_run_server_exits_when_config_invalid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(SystemExit) as excinfo:
        run_server(["--config", str(missing)])

    assert excinfo.value.code == 1
    assert "could not load configuration" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# main() dispatch
# ---------------------------------------------------------------------------


def test_main_prints_help_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["openfusion", "--help"])

    main()

    assert "Usage:" in capsys.readouterr().out


def test_main_dispatches_web_to_run_server(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(cli, "run_server", lambda argv: captured.setdefault("argv", argv))
    monkeypatch.setattr(sys, "argv", ["openfusion", "web", "--port", "9000"])

    main()

    assert captured["argv"] == ["--port", "9000"]


def test_main_dispatches_serve_alias_to_run_server(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(cli, "run_server", lambda argv: captured.setdefault("argv", argv))
    monkeypatch.setattr(sys, "argv", ["openfusion", "serve"])

    main()

    assert captured["argv"] == []


def test_main_dispatches_ask_to_ask_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(cli, "_ask_cli", lambda argv: captured.setdefault("argv", argv))
    monkeypatch.setattr(sys, "argv", ["openfusion", "ask", "hello"])

    main()

    assert captured["argv"] == ["hello"]


def test_main_dispatches_setup_to_run_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    called = []
    monkeypatch.setattr(cli, "run_setup", lambda: called.append(True))
    monkeypatch.setattr(sys, "argv", ["openfusion", "setup"])

    main()

    assert called == [True]


def test_main_dispatches_chat_to_run_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        cli,
        "run_chat",
        lambda config_path, max_tokens: captured.update(
            config_path=config_path, max_tokens=max_tokens
        ),
    )
    monkeypatch.setattr(sys, "argv", ["openfusion", "chat"])

    main()

    assert captured["max_tokens"] is None


def test_main_rejects_unknown_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["openfusion", "bogus"])

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "unknown command 'bogus'" in err
    assert "Usage:" in err


def test_main_bare_invocation_with_piped_prompt_runs_ask(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        cli,
        "run_ask",
        lambda prompt, config_path, max_tokens: captured.update(
            prompt=prompt, config_path=config_path, max_tokens=max_tokens
        ),
    )
    monkeypatch.setattr(sys, "argv", ["openfusion"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(sys.stdin, "read", lambda: "piped prompt\n")

    main()

    assert captured["prompt"] == "piped prompt"


def test_main_bare_invocation_with_empty_pipe_prints_help(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["openfusion"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(sys.stdin, "read", lambda: "   \n")

    main()

    assert "Usage:" in capsys.readouterr().out


def test_main_bare_invocation_in_a_tty_starts_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    called = []
    monkeypatch.setattr(cli, "run_chat", lambda config_path, max_tokens: called.append(True))
    monkeypatch.setattr(sys, "argv", ["openfusion"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    main()

    assert called == [True]


def _scripted(*answers: str):
    """A stand-in for input()/getpass.getpass() that returns each answer in turn,
    then raises EOFError once exhausted (mimicking Ctrl-D)."""
    remaining = iter(answers)

    def _next(_prompt: str = "") -> str:
        try:
            return next(remaining)
        except StopIteration:
            raise EOFError from None

    return _next


# ---------------------------------------------------------------------------
# run_chat: startup and API-key handling
# ---------------------------------------------------------------------------


def test_run_chat_exits_when_config_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(SystemExit) as excinfo:
        run_chat(str(missing), None)

    assert excinfo.value.code == 1
    assert "could not load configuration" in capsys.readouterr().out


def test_run_chat_exits_when_no_api_key_and_not_interactive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_saved_key", lambda: None)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    config_path = tmp_path / "openfusion.yaml"
    config_path.write_text(
        'panel:\n  - base_url: https://example.com/v1\n    api_key: ""\n    model: m\n',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        run_chat(str(config_path), None)

    assert excinfo.value.code == 1
    assert "no upstream API key" in capsys.readouterr().out


def test_run_chat_prompts_and_saves_key_when_tty_and_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_saved_key", lambda: None)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.getpass, "getpass", _scripted("sk-fresh"))
    monkeypatch.setattr("builtins.input", _scripted())  # EOFError -> exit REPL immediately
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    config_path = tmp_path / "openfusion.yaml"
    config_path.write_text(
        'panel:\n  - base_url: https://example.com/v1\n    api_key: ""\n    model: m\n',
        encoding="utf-8",
    )

    run_chat(str(config_path), None)

    out = capsys.readouterr().out
    assert "key saved" in out
    assert credentials.load_saved_key() == "sk-fresh"


# ---------------------------------------------------------------------------
# run_chat: REPL loop
# ---------------------------------------------------------------------------


def _budget_config_path(tmp_path: Path) -> Path:
    """A config that already has a valid key, so run_chat skips the key prompt."""
    config_path = tmp_path / "openfusion.yaml"
    config_path.write_text("preset: budget\n", encoding="utf-8")
    return config_path


def test_run_chat_handles_slash_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "builtins.input",
        _scripted(
            "/help",
            "",
            "/models",
            "/clear",
            "/preset quality",
            "/preset bogus",
            "/tokens 50",
            "/tokens abc",
            "/nope",
            "/quit",
        ),
    )

    run_chat(str(_budget_config_path(tmp_path)), None)

    out = capsys.readouterr().out
    assert "Commands" in out
    assert "panel:" in out and "judge:" in out
    assert "conversation cleared" in out
    assert "switched to quality" in out
    assert "usage: /preset quality|budget" in out
    assert "max tokens per call = 50" in out
    assert "usage: /tokens <n>" in out
    assert "unknown command: /nope" in out
    assert "bye" in out


def test_run_chat_key_command_updates_and_saves_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("builtins.input", _scripted("/key", "/quit"))
    monkeypatch.setattr(cli.getpass, "getpass", _scripted("sk-rotated"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

    run_chat(str(_budget_config_path(tmp_path)), None)

    out = capsys.readouterr().out
    assert "key updated and saved" in out
    assert credentials.load_saved_key() == "sk-rotated"


def test_run_chat_eof_on_prompt_exits_gracefully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("builtins.input", _scripted())

    run_chat(str(_budget_config_path(tmp_path)), None)

    assert "bye" in capsys.readouterr().out


def test_run_chat_keyboard_interrupt_on_prompt_exits_gracefully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _raise_interrupt(_prompt: str = "") -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _raise_interrupt)

    run_chat(str(_budget_config_path(tmp_path)), None)

    assert "bye" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# run_chat: a message turn (success, error, cancellation)
# ---------------------------------------------------------------------------


def test_run_chat_turn_success_continues_the_conversation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def _fake_chat_turn(_messages, _config, _console):
        return "assistant reply"

    monkeypatch.setattr(cli, "_chat_turn", _fake_chat_turn)
    monkeypatch.setattr("builtins.input", _scripted("hello there"))

    run_chat(str(_budget_config_path(tmp_path)), None)

    out = capsys.readouterr().out
    assert "[error]" not in out


def test_run_chat_turn_error_shows_key_hint_on_auth_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def _fake_chat_turn(_messages, _config, _console):
        raise RuntimeError("HTTP 401: invalid api key")

    monkeypatch.setattr(cli, "_chat_turn", _fake_chat_turn)
    monkeypatch.setattr("builtins.input", _scripted("hello"))

    run_chat(str(_budget_config_path(tmp_path)), None)

    out = capsys.readouterr().out
    assert "invalid api key" in out
    assert "may be invalid or out of credit" in out


def test_run_chat_turn_error_without_hint_for_unrelated_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def _fake_chat_turn(_messages, _config, _console):
        raise RuntimeError("boom, something else broke")

    monkeypatch.setattr(cli, "_chat_turn", _fake_chat_turn)
    monkeypatch.setattr("builtins.input", _scripted("hello"))

    run_chat(str(_budget_config_path(tmp_path)), None)

    out = capsys.readouterr().out
    assert "boom, something else broke" in out
    assert "may be invalid or out of credit" not in out


def test_run_chat_turn_keyboard_interrupt_cancels_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def _fake_chat_turn(_messages, _config, _console):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_chat_turn", _fake_chat_turn)
    monkeypatch.setattr("builtins.input", _scripted("hello"))

    run_chat(str(_budget_config_path(tmp_path)), None)

    assert "cancelled" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# run_setup
# ---------------------------------------------------------------------------


def test_run_setup_declines_overwrite_keeps_existing_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    existing = Path("openfusion.yaml")
    existing.write_text("preset: budget\n", encoding="utf-8")
    monkeypatch.setattr("builtins.input", _scripted("n"))

    run_setup()

    assert existing.read_text(encoding="utf-8") == "preset: budget\n"
    assert "Keeping your existing config" in capsys.readouterr().out


def test_run_setup_empty_key_bails_without_writing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.getpass, "getpass", _scripted(""))

    run_setup()

    assert not Path("openfusion.yaml").exists()
    assert "No key entered" in capsys.readouterr().out


def test_run_setup_writes_budget_preset_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.getpass, "getpass", _scripted("sk-new"))
    monkeypatch.setattr("builtins.input", _scripted(""))  # blank -> default preset (budget)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

    run_setup()

    written = Path("openfusion.yaml")
    config = load_config(written)
    assert config.preset is None  # build_setup_yaml expands the preset inline
    assert all(member.api_key == "sk-new" for member in config.panel)
    assert config.judge is not None and config.judge.model == "deepseek/deepseek-v4-pro"
    assert credentials.load_saved_key() == "sk-new"
    if os.name == "posix":
        import stat

        mode = stat.S_IMODE(written.stat().st_mode)
        assert mode == 0o600
    assert "budget preset" in capsys.readouterr().out


def test_run_setup_writes_quality_preset_when_chosen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.getpass, "getpass", _scripted("sk-new"))
    monkeypatch.setattr("builtins.input", _scripted("2"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

    run_setup()

    config = load_config(Path("openfusion.yaml"))
    assert config.judge is not None and config.judge.model == "anthropic/claude-sonnet-4"
    assert "quality preset" in capsys.readouterr().out


def test_run_setup_overwrite_accepted_replaces_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("openfusion.yaml").write_text("preset: budget\n", encoding="utf-8")
    monkeypatch.setattr("builtins.input", _scripted("y", ""))
    monkeypatch.setattr(cli.getpass, "getpass", _scripted("sk-overwritten"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

    run_setup()

    config = load_config(Path("openfusion.yaml"))
    assert all(member.api_key == "sk-overwritten" for member in config.panel)
