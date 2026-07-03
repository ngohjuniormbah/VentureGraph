"""
Generic asynchronous ReAct-style tool-use loop for Claude.

The Venture Catalyst's Market Intelligence and TAM Calculator agents both
need Claude to autonomously decide *when* to call a tool (web search,
Python execution), read the result, and decide whether to call another
tool or give a final answer. That's a multi-turn pattern Instructor's
single-shot `response_model` extraction isn't built for (it's designed for
"one call in, one validated object out"). This module implements that
multi-turn loop directly against the Anthropic tool-use API; Instructor is
then used *downstream* of this loop's plain-text final answer, to coerce it
into a strict Pydantic schema (see `src.agents.venture_catalyst`).
"""

import dataclasses
from typing import Any, Awaitable, Callable

from anthropic import AsyncAnthropic


@dataclasses.dataclass
class ToolSpec:
    """
    Definition of one tool Claude may call during an agentic loop.

    Attributes:
        name: Tool name, as Claude will refer to it.
        description: What the tool does and when to use it - shown to
            Claude verbatim, so this is effectively part of the prompt.
        input_schema: JSON schema describing the tool's arguments.
        executor: An async function taking the tool call's arguments as
            keyword arguments and returning a string to send back to
            Claude as the tool's result.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    executor: Callable[..., Awaitable[str]]

    def to_anthropic_tool(self) -> dict[str, Any]:
        """Render this spec as the tool-definition dict the Anthropic API expects."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


async def run_agentic_tool_loop(
    client: AsyncAnthropic,
    system: str,
    user_message: str,
    tools: list[ToolSpec],
    model: str,
    max_turns: int = 6,
    max_tokens: int = 2048,
) -> str:
    """
    Run a multi-turn tool-use conversation until Claude gives a final answer.

    Data flow:
        1. Sends `user_message` to Claude along with the given `tools`
           (converted to Anthropic's tool-definition format).
        2. If Claude's response has `stop_reason == "tool_use"`, executes
           every requested tool call by looking up its `executor` in
           `tools` and awaiting it with the model-provided arguments, then
           appends both Claude's tool-call message and a `tool_result`
           message (one per call) back onto the conversation.
        3. Repeats step 1-2 - so a tool's output can itself prompt another
           tool call (e.g. search, read the results, search again with a
           refined query) - until Claude responds without requesting a
           tool, or `max_turns` is reached.
        4. Returns the concatenated text of that final response, which the
           caller typically passes to a downstream Instructor call to
           extract structured data from it.

    Args:
        client: An `AsyncAnthropic` client (not Instructor-wrapped - this
            loop needs raw access to `tools`/`tool_use` semantics).
        system: System prompt for the conversation.
        user_message: The initial user turn.
        tools: The tools Claude is allowed to call this turn.
        model: Anthropic model id to use.
        max_turns: Maximum number of request/tool-call round trips before
            giving up.
        max_tokens: Max tokens per Claude response.

    Returns:
        Claude's final free-text answer (the text content of the first
        response that doesn't request a tool call).

    Raises:
        RuntimeError: If `max_turns` is exceeded without Claude producing a
            final (non-tool-use) answer.
    """
    tool_defs = [tool.to_anthropic_tool() for tool in tools]
    executors = {tool.name: tool.executor for tool in tools}
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

    for _ in range(max_turns):
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tool_defs,
        )

        if response.stop_reason != "tool_use":
            return "".join(block.text for block in response.content if block.type == "text")

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                executor = executors[block.name]
                try:
                    result_text = await executor(**block.input)
                except Exception as exc:  # surfaced to the model as a tool error, not a crash
                    result_text = f"Tool error: {exc}"
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result_text}
                )
        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(f"Tool loop exceeded max_turns={max_turns} without a final answer.")
