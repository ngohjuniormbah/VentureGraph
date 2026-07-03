"""
Tests for the generic, provider-agnostic agentic tool-use loop.

These use a minimal fake `ChatAdapter` (implementing just `send()` /
`record_tool_results()`) rather than a fake provider client, since
`run_agentic_tool_loop` itself never touches provider wire formats -
that's exactly the point of the `ChatAdapter` seam (see
`src/agents/chat_adapters.py` for the real Anthropic/Gemini
implementations, and `test_chat_adapters.py` for tests of those).
"""

import pytest

from src.agents.tool_loop import ChatTurn, ToolCall, ToolSpec, run_agentic_tool_loop


class _FakeAdapter:
    """Returns a scripted sequence of ChatTurns, one per `send()` call."""

    def __init__(self, turns: list[ChatTurn]):
        self._turns = list(turns)
        self.recorded_results: list[list[tuple[ToolCall, str]]] = []

    async def send(self) -> ChatTurn:
        return self._turns.pop(0)

    def record_tool_results(self, results: list[tuple[ToolCall, str]]) -> None:
        self.recorded_results.append(results)


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

    adapter = _FakeAdapter(
        turns=[
            ChatTurn(text=None, tool_calls=[ToolCall(id="call_1", name="web_search", input={"query": "widget competitors"})]),
            ChatTurn(text="Acme Corp is the main competitor.", tool_calls=[]),
        ]
    )

    result = await run_agentic_tool_loop(adapter, tools=[tool])

    assert result == "Acme Corp is the main competitor."
    assert calls_made == ["widget competitors"]
    assert len(adapter.recorded_results) == 1


async def test_loop_raises_after_max_turns_without_final_answer():
    async def fake_search(query: str) -> str:
        return "still searching"

    tool = ToolSpec(
        name="web_search",
        description="search the web",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        executor=fake_search,
    )

    # Every turn requests another tool call - never a final answer.
    endless_tool_use = ChatTurn(text=None, tool_calls=[ToolCall(id="call_n", name="web_search", input={"query": "q"})])
    adapter = _FakeAdapter(turns=[endless_tool_use] * 3)

    with pytest.raises(RuntimeError):
        await run_agentic_tool_loop(adapter, tools=[tool], max_turns=3)


async def test_tool_executor_exception_is_surfaced_as_tool_error_not_a_crash():
    async def failing_tool(query: str) -> str:
        raise ValueError("boom")

    tool = ToolSpec(
        name="web_search",
        description="search the web",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        executor=failing_tool,
    )

    adapter = _FakeAdapter(
        turns=[
            ChatTurn(text=None, tool_calls=[ToolCall(id="call_1", name="web_search", input={"query": "q"})]),
            ChatTurn(text="recovered", tool_calls=[]),
        ]
    )

    result = await run_agentic_tool_loop(adapter, tools=[tool])

    assert result == "recovered"
    ((call, result_text),) = adapter.recorded_results[0]
    assert "Tool error: boom" in result_text
