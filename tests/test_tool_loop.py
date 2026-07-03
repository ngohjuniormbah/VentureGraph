"""
Tests for the generic agentic tool-use loop, using a fake Anthropic client
(no real API call, no network) that simulates: tool call -> tool result ->
final answer.
"""

import types

import pytest

from src.agents.tool_loop import ToolSpec, run_agentic_tool_loop


def _text_block(text: str):
    return types.SimpleNamespace(type="text", text=text)


def _tool_use_block(tool_use_id: str, name: str, tool_input: dict):
    return types.SimpleNamespace(type="tool_use", id=tool_use_id, name=name, input=tool_input)


def _response(stop_reason: str, content: list):
    return types.SimpleNamespace(stop_reason=stop_reason, content=content)


class _FakeMessages:
    """Returns a scripted sequence of responses, one per call, ignoring kwargs."""

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses: list):
        self.messages = _FakeMessages(responses)


@pytest.mark.asyncio
async def test_loop_calls_tool_then_returns_final_text():
    calls_made = []

    async def fake_search(query: str) -> str:
        calls_made.append(query)
        return "Result: Acme Corp is a competitor."

    tool = ToolSpec(
        name="web_search",
        description="search the web",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        executor=fake_search,
    )

    fake_client = _FakeClient(
        responses=[
            _response("tool_use", [_tool_use_block("call_1", "web_search", {"query": "widget competitors"})]),
            _response("end_turn", [_text_block("Acme Corp is the main competitor.")]),
        ]
    )

    result = await run_agentic_tool_loop(
        fake_client, system="you are helpful", user_message="find competitors", tools=[tool], model="fake-model"
    )

    assert result == "Acme Corp is the main competitor."
    assert calls_made == ["widget competitors"]
    # Two round trips: the tool-use turn, then the final-answer turn.
    assert len(fake_client.messages.calls) == 2


@pytest.mark.asyncio
async def test_loop_raises_after_max_turns_without_final_answer():
    async def fake_search(query: str) -> str:
        return "still searching"

    tool = ToolSpec(
        name="web_search",
        description="search the web",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        executor=fake_search,
    )

    # Every response requests another tool call - never a final answer.
    endless_tool_use = _response("tool_use", [_tool_use_block("call_n", "web_search", {"query": "q"})])
    fake_client = _FakeClient(responses=[endless_tool_use] * 3)

    with pytest.raises(RuntimeError):
        await run_agentic_tool_loop(
            fake_client, system="s", user_message="u", tools=[tool], model="fake-model", max_turns=3
        )


@pytest.mark.asyncio
async def test_tool_executor_exception_is_surfaced_as_tool_error_not_a_crash():
    async def failing_tool(query: str) -> str:
        raise ValueError("boom")

    tool = ToolSpec(
        name="web_search",
        description="search the web",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        executor=failing_tool,
    )

    fake_client = _FakeClient(
        responses=[
            _response("tool_use", [_tool_use_block("call_1", "web_search", {"query": "q"})]),
            _response("end_turn", [_text_block("recovered")]),
        ]
    )

    result = await run_agentic_tool_loop(
        fake_client, system="s", user_message="u", tools=[tool], model="fake-model"
    )

    assert result == "recovered"
    # The second call's messages should contain the tool error, not raise.
    second_call_messages = fake_client.messages.calls[1]["messages"]
    tool_result_content = second_call_messages[-1]["content"][0]["content"]
    assert "Tool error: boom" in tool_result_content
