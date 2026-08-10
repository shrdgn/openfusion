"""Tests for the sequential pipeline strategy."""

from __future__ import annotations

import httpx
import pytest
import respx

from openfusion.config import (
    JudgeConfig,
    OpenFusionConfig,
    PanelMember,
    PassThroughConfig,
    PipelineConfig,
    PipelineStepConfig,
    PipelineStepUse,
    Strategy,
)
from openfusion.errors import InvalidRequestError, UpstreamError
from openfusion.pipeline import (
    _build_step_messages,
    _collect_stream,
    _inject_step_outputs,
    run_pipeline,
)
from openfusion.upstream import UpstreamClient

_SSE_HELLO = (
    'data: {"choices":[{"delta":{"role":"assistant","content":"hel"}'
    ',"finish_reason":null}]}\n\n'
    'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}\n\n'
    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)


def _sse_response(text: str = _SSE_HELLO) -> httpx.Response:
    return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})


def _base_config(steps: list[PipelineStepConfig]) -> OpenFusionConfig:
    return OpenFusionConfig(
        strategy=Strategy.PIPELINE,
        pipeline=PipelineConfig(steps=steps),
        panel=[PanelMember(base_url="https://mock/v1", api_key="k", model="m")],
        judge=JudgeConfig(base_url="https://mock/v1", api_key="k", model="j"),
        pass_through=PassThroughConfig(base_url="https://mock/v1", api_key="k", model="m"),
    )

# ---------------------------------------------------------------------------
# _inject_step_outputs
# ---------------------------------------------------------------------------


def test_inject_replaces_placeholder() -> None:
    result = _inject_step_outputs("Summary: {research}", {"research": "lots of facts"})
    assert result == "Summary: lots of facts"


def test_inject_multiple_placeholders() -> None:
    result = _inject_step_outputs(
        "{research} | {critique}",
        {"research": "R", "critique": "C"},
    )
    assert result == "R | C"


def test_inject_missing_placeholder_left_intact() -> None:
    result = _inject_step_outputs("hello {missing}", {"research": "R"})
    assert result == "hello {missing}"


def test_inject_empty_outputs() -> None:
    result = _inject_step_outputs("plain text", {})
    assert result == "plain text"


# ---------------------------------------------------------------------------
# _build_step_messages
# ---------------------------------------------------------------------------


def test_build_no_system_no_outputs() -> None:
    msgs = [{"role": "user", "content": "hi"}]
    result = _build_step_messages(msgs, None, {})
    assert result == [{"role": "user", "content": "hi"}]


def test_build_strips_original_system() -> None:
    msgs = [
        {"role": "system", "content": "original"},
        {"role": "user", "content": "hi"},
    ]
    result = _build_step_messages(msgs, "new system", {})
    assert result[0] == {"role": "system", "content": "new system"}
    assert result[1] == {"role": "user", "content": "hi"}
    assert len(result) == 2


def test_build_injects_outputs_into_system() -> None:
    msgs = [{"role": "user", "content": "q"}]
    result = _build_step_messages(msgs, "Use {step1} here", {"step1": "answer1"})
    assert result[0]["content"] == "Use answer1 here"


def test_build_no_system_with_outputs_adds_context_block() -> None:
    msgs = [{"role": "user", "content": "q"}]
    result = _build_step_messages(msgs, None, {"step1": "prior output"})
    assert result[0]["role"] == "system"
    assert "prior output" in result[0]["content"]
    assert result[1] == {"role": "user", "content": "q"}


# ---------------------------------------------------------------------------
# PipelineStepConfig defaults
# ---------------------------------------------------------------------------


def test_step_default_use_is_solo() -> None:
    step = PipelineStepConfig(name="s1")
    assert step.use == PipelineStepUse.SOLO


def test_step_explicit_fuse() -> None:
    step = PipelineStepConfig(name="s1", use=PipelineStepUse.FUSE)
    assert step.use == PipelineStepUse.FUSE


def test_step_model_override() -> None:
    step = PipelineStepConfig(name="s1", model="gpt-4o-mini")
    assert step.model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# run_pipeline: empty steps raises
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_pipeline_empty_steps_raises() -> None:
    from openfusion.config import (
        JudgeConfig,
        OpenFusionConfig,
        PanelMember,
        PassThroughConfig,
        PipelineConfig,
        Strategy,
    )
    from openfusion.errors import InvalidRequestError
    from openfusion.pipeline import run_pipeline
    from openfusion.upstream import UpstreamClient

    cfg = OpenFusionConfig(
        strategy=Strategy.PIPELINE,
        pipeline=PipelineConfig(steps=[]),
        panel=[PanelMember(base_url="https://mock/v1", api_key="k", model="m")],
        judge=JudgeConfig(base_url="https://mock/v1", api_key="k", model="j"),
        pass_through=PassThroughConfig(base_url="https://mock/v1", api_key="k", model="m"),
    )
    client = UpstreamClient()
    body = {"messages": [{"role": "user", "content": "hi"}]}

    gen = run_pipeline(body, cfg, client)
    with pytest.raises(InvalidRequestError, match="no steps"):
        async for _ in gen:
            pass
    await client.aclose()


# ---------------------------------------------------------------------------
# run_pipeline: single SOLO step (mocked upstream)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_pipeline_single_solo_step() -> None:
    import httpx
    import respx

    from openfusion.config import (
        JudgeConfig,
        OpenFusionConfig,
        PanelMember,
        PassThroughConfig,
        PipelineConfig,
        PipelineStepConfig,
        PipelineStepUse,
        Strategy,
    )
    from openfusion.pipeline import run_pipeline
    from openfusion.upstream import UpstreamClient

    cfg = OpenFusionConfig(
        strategy=Strategy.PIPELINE,
        pipeline=PipelineConfig(
            steps=[PipelineStepConfig(name="answer", use=PipelineStepUse.SOLO)]
        ),
        panel=[PanelMember(base_url="https://mock/v1", api_key="k", model="m")],
        judge=JudgeConfig(base_url="https://mock/v1", api_key="k", model="j"),
        pass_through=PassThroughConfig(base_url="https://mock/v1", api_key="k", model="m"),
    )

    sse_body = (
        'data: {"choices":[{"delta":{"role":"assistant","content":"hello"}'
        ',"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )

    with respx.mock(assert_all_called=True) as mock:
        mock.post("https://mock/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, text=sse_body, headers={"content-type": "text/event-stream"}
            )
        )
        client = UpstreamClient()
        parts: list[str] = []
        async for delta, _usage, _reason in run_pipeline(
            {"messages": [{"role": "user", "content": "hi"}]}, cfg, client
        ):
            if delta:
                parts.append(delta)
        await client.aclose()

    assert "".join(parts) == "hello"


# ---------------------------------------------------------------------------
# _collect_stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_stream_accumulates_text_and_last_usage() -> None:
    async def chunks():
        yield {"choices": [{"delta": {"content": "foo"}}]}
        yield {"choices": [{"delta": {"content": "bar"}}]}
        yield {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }

    text, usage = await _collect_stream(chunks())
    assert text == "foobar"
    assert usage == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}


@pytest.mark.asyncio
async def test_collect_stream_no_usage_returns_none() -> None:
    async def chunks():
        yield {"choices": [{"delta": {"content": "x"}}]}

    text, usage = await _collect_stream(chunks())
    assert text == "x"
    assert usage is None


# ---------------------------------------------------------------------------
# run_pipeline: FUSE steps
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_pipeline_fuse_step_missing_judge_raises() -> None:
    cfg = _base_config([PipelineStepConfig(name="answer", use=PipelineStepUse.FUSE)])
    cfg = cfg.model_copy(update={"judge": None})
    client = UpstreamClient()

    gen = run_pipeline({"messages": [{"role": "user", "content": "hi"}]}, cfg, client)
    with pytest.raises(InvalidRequestError, match="fuse but no judge"):
        async for _ in gen:
            pass
    await client.aclose()


@pytest.mark.asyncio
async def test_run_pipeline_single_fuse_step_streams_synthesis() -> None:
    cfg = _base_config([PipelineStepConfig(name="answer", use=PipelineStepUse.FUSE)])

    with respx.mock(assert_all_called=True) as mock:
        route = mock.post("https://mock/v1/chat/completions")
        route.side_effect = [
            # panel member call (non-streaming)
            httpx.Response(200, json={"choices": [{"message": {"content": "panel says x"}}]}),
            # judge synthesis call (streaming)
            _sse_response(),
        ]
        client = UpstreamClient()
        parts: list[str] = []
        async for delta, _usage, _reason in run_pipeline(
            {"messages": [{"role": "user", "content": "hi"}]}, cfg, client
        ):
            if delta:
                parts.append(delta)
        await client.aclose()

    assert "".join(parts) == "hello"


@pytest.mark.asyncio
async def test_run_pipeline_intermediate_fuse_feeds_final_solo() -> None:
    """An intermediate FUSE step's synthesized text must be injected into the
    next step's {step_name} placeholder, and only the final step's output is
    streamed to the caller."""
    cfg = _base_config(
        [
            PipelineStepConfig(name="research", use=PipelineStepUse.FUSE),
            PipelineStepConfig(
                name="final",
                use=PipelineStepUse.SOLO,
                system="Use {research} to answer.",
            ),
        ]
    )

    with respx.mock(assert_all_called=True) as mock:
        route = mock.post("https://mock/v1/chat/completions")
        route.side_effect = [
            httpx.Response(200, json={"choices": [{"message": {"content": "panel says x"}}]}),
            _sse_response('data: {"choices":[{"delta":{"content":"research done"}'
                          ',"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'),
            _sse_response('data: {"choices":[{"delta":{"content":"final answer"}'
                          ',"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'),
        ]
        client = UpstreamClient()
        parts: list[str] = []
        async for delta, _usage, _reason in run_pipeline(
            {"messages": [{"role": "user", "content": "hi"}]}, cfg, client
        ):
            if delta:
                parts.append(delta)
        await client.aclose()

        # The final step's system prompt must have had the research output injected.
        final_call = route.calls[-1]
        sent_body = final_call.request.content.decode()
        assert "research done" in sent_body

    assert "".join(parts) == "final answer"


# ---------------------------------------------------------------------------
# run_pipeline: SOLO steps
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_pipeline_solo_step_model_override() -> None:
    cfg = _base_config(
        [PipelineStepConfig(name="answer", use=PipelineStepUse.SOLO, model="custom/model")]
    )

    with respx.mock(assert_all_called=True) as mock:
        route = mock.post("https://mock/v1/chat/completions").mock(
            return_value=_sse_response()
        )
        client = UpstreamClient()
        parts: list[str] = []
        async for delta, _usage, _reason in run_pipeline(
            {"messages": [{"role": "user", "content": "hi"}]}, cfg, client
        ):
            if delta:
                parts.append(delta)
        await client.aclose()

        sent_body = route.calls[-1].request.content.decode()
        assert '"model":"custom/model"' in sent_body or "custom/model" in sent_body

    assert "".join(parts) == "hello"


@pytest.mark.asyncio
async def test_run_pipeline_multi_step_solo_chain() -> None:
    """Two SOLO steps chained: the first's collected output must be injected
    into the second's system prompt via the {step_name} placeholder."""
    cfg = _base_config(
        [
            PipelineStepConfig(name="draft", use=PipelineStepUse.SOLO),
            PipelineStepConfig(
                name="final", use=PipelineStepUse.SOLO, system="Polish: {draft}"
            ),
        ]
    )

    with respx.mock(assert_all_called=True) as mock:
        route = mock.post("https://mock/v1/chat/completions")
        route.side_effect = [
            _sse_response('data: {"choices":[{"delta":{"content":"draft text"}'
                          ',"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'),
            _sse_response('data: {"choices":[{"delta":{"content":"polished"}'
                          ',"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'),
        ]
        client = UpstreamClient()
        parts: list[str] = []
        async for delta, _usage, _reason in run_pipeline(
            {"messages": [{"role": "user", "content": "hi"}]}, cfg, client
        ):
            if delta:
                parts.append(delta)
        await client.aclose()

        final_call = route.calls[-1]
        assert "draft text" in final_call.request.content.decode()

    assert "".join(parts) == "polished"


@pytest.mark.asyncio
async def test_run_pipeline_multi_step_accumulates_usage_across_steps() -> None:
    """Every step's usage must reach the caller, not just the final step's.

    Regression test: intermediate steps used to discard their usage entirely
    (bound to a `_usage`/`_reason` throwaway variable), so a pipeline's reported
    cost silently covered only its last step.
    """
    cfg = _base_config(
        [
            PipelineStepConfig(name="draft", use=PipelineStepUse.SOLO),
            PipelineStepConfig(
                name="final", use=PipelineStepUse.SOLO, system="Polish: {draft}"
            ),
        ]
    )

    draft_sse = (
        'data: {"choices":[{"delta":{"content":"draft text"},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"cost":0.01}}\n\n'
        "data: [DONE]\n\n"
    )
    final_sse = (
        'data: {"choices":[{"delta":{"content":"polished"},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":20,"completion_tokens":8,"total_tokens":28,"cost":0.02}}\n\n'
        "data: [DONE]\n\n"
    )

    with respx.mock(assert_all_called=True) as mock:
        route = mock.post("https://mock/v1/chat/completions")
        route.side_effect = [_sse_response(draft_sse), _sse_response(final_sse)]
        client = UpstreamClient()
        last_usage = None
        async for _delta, usage, _reason in run_pipeline(
            {"messages": [{"role": "user", "content": "hi"}]}, cfg, client
        ):
            if usage:
                last_usage = usage
        await client.aclose()

    assert last_usage == {
        "prompt_tokens": 30,
        "completion_tokens": 13,
        "total_tokens": 43,
        "cost": 0.03,
    }


@pytest.mark.asyncio
async def test_run_pipeline_final_fuse_step_folds_in_panel_and_prior_usage() -> None:
    """A FUSE step's own panel usage -- and any accumulated from earlier steps --
    must reach the caller alongside the judge's, even when FUSE is the final step.
    synthesize() only ever yields the judge's usage, so run_pipeline has to fold in
    the panel usage itself.
    """
    cfg = _base_config(
        [
            PipelineStepConfig(name="draft", use=PipelineStepUse.SOLO),
            PipelineStepConfig(name="final", use=PipelineStepUse.FUSE),
        ]
    )

    draft_sse = (
        'data: {"choices":[{"delta":{"content":"draft text"},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\n'
        "data: [DONE]\n\n"
    )
    panel_json = {
        "choices": [{"message": {"content": "panel says x"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    }
    judge_sse = (
        'data: {"choices":[{"delta":{"content":"final answer"},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":20, "completion_tokens":1,"total_tokens":21}}\n\n'
        "data: [DONE]\n\n"
    )

    with respx.mock(assert_all_called=True) as mock:
        route = mock.post("https://mock/v1/chat/completions")
        route.side_effect = [
            _sse_response(draft_sse),  # SOLO draft step
            httpx.Response(200, json=panel_json),  # FUSE step's panel member
            _sse_response(judge_sse),  # FUSE step's judge synthesis
        ]
        client = UpstreamClient()
        last_usage = None
        async for _delta, usage, _reason in run_pipeline(
            {"messages": [{"role": "user", "content": "hi"}]}, cfg, client
        ):
            if usage:
                last_usage = usage
        await client.aclose()

    # draft (5/2/7) + panel (10/3/13) + judge (20/1/21) = 35/6/41
    assert last_usage == {
        "prompt_tokens": 35,
        "completion_tokens": 6,
        "total_tokens": 41,
    }


@pytest.mark.asyncio
async def test_run_pipeline_intermediate_fuse_step_usage_is_accumulated() -> None:
    """An intermediate FUSE step's panel usage AND its own judge synthesis usage must
    both be folded into the final total, not just the final step's usage."""
    cfg = _base_config(
        [
            PipelineStepConfig(name="research", use=PipelineStepUse.FUSE),
            PipelineStepConfig(
                name="final", use=PipelineStepUse.SOLO, system="Use {research} to answer."
            ),
        ]
    )

    panel_json = {
        "choices": [{"message": {"content": "panel says x"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    }
    research_sse = (
        'data: {"choices":[{"delta":{"content":"research done"},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":20,"completion_tokens":1,"total_tokens":21}}\n\n'
        "data: [DONE]\n\n"
    )
    final_sse = (
        'data: {"choices":[{"delta":{"content":"final answer"},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\n'
        "data: [DONE]\n\n"
    )

    with respx.mock(assert_all_called=True) as mock:
        route = mock.post("https://mock/v1/chat/completions")
        route.side_effect = [
            httpx.Response(200, json=panel_json),  # research step's panel member
            _sse_response(research_sse),  # research step's judge synthesis
            _sse_response(final_sse),  # final SOLO step
        ]
        client = UpstreamClient()
        last_usage = None
        async for _delta, usage, _reason in run_pipeline(
            {"messages": [{"role": "user", "content": "hi"}]}, cfg, client
        ):
            if usage:
                last_usage = usage
        await client.aclose()

    # panel (10/3/13) + judge (20/1/21) + final solo (5/2/7) = 35/6/41
    assert last_usage == {
        "prompt_tokens": 35,
        "completion_tokens": 6,
        "total_tokens": 41,
    }


@pytest.mark.asyncio
async def test_run_pipeline_emits_accumulated_usage_when_final_step_has_none() -> None:
    """If the final step's own stream never carries a usage field, prior steps'
    accumulated usage must still reach the caller via a trailing marker instead of
    being silently dropped."""
    cfg = _base_config(
        [
            PipelineStepConfig(name="draft", use=PipelineStepUse.SOLO),
            PipelineStepConfig(name="final", use=PipelineStepUse.SOLO),
        ]
    )

    draft_sse = (
        'data: {"choices":[{"delta":{"content":"draft text"},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\n'
        "data: [DONE]\n\n"
    )
    final_sse_no_usage = (
        'data: {"choices":[{"delta":{"content":"final answer"},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )

    with respx.mock(assert_all_called=True) as mock:
        route = mock.post("https://mock/v1/chat/completions")
        route.side_effect = [_sse_response(draft_sse), _sse_response(final_sse_no_usage)]
        client = UpstreamClient()
        last_usage = None
        async for _delta, usage, _reason in run_pipeline(
            {"messages": [{"role": "user", "content": "hi"}]}, cfg, client
        ):
            if usage:
                last_usage = usage
        await client.aclose()

    assert last_usage == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}


@pytest.mark.asyncio
async def test_run_pipeline_solo_final_step_non_stream_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _base_config([PipelineStepConfig(name="answer", use=PipelineStepUse.SOLO)])
    client = UpstreamClient()

    async def fake_chat_completion(*args, **kwargs):
        return {"choices": [{"message": {"content": "not a stream"}}]}

    monkeypatch.setattr(client, "chat_completion", fake_chat_completion)

    gen = run_pipeline({"messages": [{"role": "user", "content": "hi"}]}, cfg, client)
    with pytest.raises(UpstreamError, match="Expected streaming response"):
        async for _ in gen:
            pass
    await client.aclose()


@pytest.mark.asyncio
async def test_run_pipeline_solo_intermediate_step_non_stream_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _base_config(
        [
            PipelineStepConfig(name="draft", use=PipelineStepUse.SOLO),
            PipelineStepConfig(name="final", use=PipelineStepUse.SOLO),
        ]
    )
    client = UpstreamClient()

    async def fake_chat_completion(*args, **kwargs):
        return {"choices": [{"message": {"content": "not a stream"}}]}

    monkeypatch.setattr(client, "chat_completion", fake_chat_completion)

    gen = run_pipeline({"messages": [{"role": "user", "content": "hi"}]}, cfg, client)
    with pytest.raises(UpstreamError, match="Expected streaming response"):
        async for _ in gen:
            pass
    await client.aclose()
