"""
Generic asynchronous ReAct-style tool-use loop.

The Venture Catalyst's Market Intelligence and TAM Calculator agents both
need the model to autonomously decide *when* to call a tool (web search,
Python execution), read the result, and decide whether to call another
tool or give a final answer. That's a multi-turn pattern Instructor's
single-shot `response_model` extraction isn't built for.

This module itself has no idea which LLM provider it's talking to - it's
written entirely against the `ChatAdapter` protocol below. Each provider's
actual wire format (Anthropic's `tool_use`/`tool_result` content blocks vs.
Gemini's `function_call`/`function_response` parts) lives in
`src/agents/chat_adapters.py`, behind that same interface. That split is
what makes `LLM_PROVIDER=anthropic|gemini` (see `src/agents/llm_client.py`)
a config change instead of a rewrite: this loop, and the agents that call
it (`src/agents/venture_catalyst.py`), never change when the provider does.
"""

import dataclasses
from typing import Any, Awaitable, Callable, Protocol


@dataclasses.dataclass
class ToolSpec:
    """
    Definition of one tool the model may call during an agentic loop.

    Attributes:
        name: Tool name, as the model will refer to it.
        description: What the tool does and when to use it - shown to the
            model verbatim, so this is effectively part of the prompt.
        input_schema: JSON schema describing the tool's arguments. Passed
            to whichever provider's adapter is in use; both Anthropic's
            `tools[].input_schema` and Gemini's
            `FunctionDeclaration.parameters_json_schema` accept this same
            plain JSON-schema shape, so no per-provider translation is
            needed here.
        executor: An async function taking the tool call's arguments as
            keyword arguments and returning a string to send back to the
            model as the tool's result.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    executor: Callable[..., Awaitable[str]]


@dataclasses.dataclass
class ToolCall:
    """One tool invocation the model requested, normalized across providers."""

    id: str
    name: str
    input: dict[str, Any]


@dataclasses.dataclass
class ChatTurn:
    """
    One normalized turn from the model.

    Exactly one of `text`/`tool_calls` is meaningful: if `tool_calls` is
    non-empty, the model wants those tools run before it will continue: if
    it's empty, `text` is the model's final answer.
    """

    text: str | None
    tool_calls: list[ToolCall]


class ChatAdapter(Protocol):
    """
    Provider-specific conversation driver, used by `run_agentic_tool_loop`.

    An adapter owns the running conversation history in its provider's
    native format (constructed with the system prompt, user message, and
    tool definitions up front - see `AnthropicChatAdapter`/
    `GeminiChatAdapter` in `chat_adapters.py`), and exposes only these two
    provider-agnostic operations to the loop.
    """

    async def send(self) -> ChatTurn:
        """Send the conversation so far and return the model's next turn."""
        ...

    def record_tool_results(self, results: list[tuple[ToolCall, str]]) -> None:
        """Append executed tool results to the conversation, in whatever shape this provider expects."""
        ...


async def run_agentic_tool_loop(adapter: ChatAdapter, tools: list[ToolSpec], max_turns: int = 6) -> str:
    """
    Run a multi-turn tool-use conversation until the model gives a final answer.

    Data flow:
        1. Calls `adapter.send()`, which sends the conversation so far
           (already primed with the system prompt, user message, and tool
           definitions when the adapter was constructed) and returns a
           normalized `ChatTurn`.
        2. If `turn.tool_calls` is empty, `turn.text` is the model's final
           answer - return it.
        3. Otherwise, look up each requested tool call's `executor` (by
           name, in `tools`) and await it with the model-provided
           arguments; a failing executor produces a `"Tool error: ..."`
           string instead of raising, so one bad tool call doesn't crash
           the whole pipeline.
        4. Calls `adapter.record_tool_results()` with the
           `(ToolCall, result_text)` pairs, so the adapter can append them
           to its history in its provider's native format, then repeats
           from step 1 - so a tool's output can itself prompt another tool
           call (e.g. search, read the results, search again with a
           refined query) - until the model responds without requesting a
           tool, or `max_turns` is reached.

    Args:
        adapter: A `ChatAdapter` already primed with the system prompt,
            user message, and tool definitions (see
            `src.agents.llm_client.get_chat_adapter`).
        tools: The same tools the adapter was constructed with - used here
            only to look up each tool's `executor` by name.
        max_turns: Maximum number of request/tool-call round trips before
            giving up.

    Returns:
        The model's final free-text answer.

    Raises:
        RuntimeError: If `max_turns` is exceeded without the model
            producing a final (non-tool-call) answer.
    """
    executors = {tool.name: tool.executor for tool in tools}

    for _ in range(max_turns):
        turn = await adapter.send()

        if not turn.tool_calls:
            return turn.text or ""

        results: list[tuple[ToolCall, str]] = []
        for call in turn.tool_calls:
            executor = executors[call.name]
            try:
                result_text = await executor(**call.input)
            except Exception as exc:  # surfaced to the model as a tool error, not a crash
                result_text = f"Tool error: {exc}"
            results.append((call, result_text))

        adapter.record_tool_results(results)

    raise RuntimeError(f"Tool loop exceeded max_turns={max_turns} without a final answer.")
