"""
Provider-specific chat adapters for the agentic tool-use loop.

`run_agentic_tool_loop` (`tool_loop.py`) only knows the `ChatAdapter`
protocol - `send()` and `record_tool_results()`. Each class here owns the
actual wire format for one provider (which API call to make, how to read
a tool call out of the response, how to write a tool result back into the
conversation) and the running conversation history, so switching providers
(`LLM_PROVIDER=anthropic|gemini`, see `src/agents/llm_client.py`) never
touches the loop itself or the agents that drive it
(`src/agents/venture_catalyst.py`).

Note on Gemini coverage: `AnthropicChatAdapter` is exercised by this
project's original agents and tests against the real Anthropic API shape.
`GeminiChatAdapter` is implemented directly against the documented
`google-genai` request/response schema (verified by introspecting the
installed SDK and by a dry-run call that reached Google's real API and
failed only on an invalid placeholder key - i.e. the request shape itself
was accepted), but has not been exercised against a live Gemini
conversation with real tool calls, since no Gemini API key was available
in the environment this was built in. If you hit a mismatch, the most
likely spot is `FunctionCall.id` handling (Gemini doesn't always populate
it the way the streaming/live API does for standard `generateContent`
calls) - see the fallback to `fc.name` below.
"""

from typing import Any

from anthropic import AsyncAnthropic

from src.agents.tool_loop import ChatTurn, ToolCall, ToolSpec


class AnthropicChatAdapter:
    """Drives a tool-use conversation against the Anthropic Messages API."""

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        system: str,
        user_message: str,
        tools: list[ToolSpec],
        max_tokens: int = 2048,
    ):
        self._client = client
        self._model = model
        self._system = system
        self._max_tokens = max_tokens
        self._tool_defs = [
            {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
            for tool in tools
        ]
        self._messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

    async def send(self) -> ChatTurn:
        """Call `messages.create` with the tool definitions and history built so far, and normalize the reply."""
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system,
            messages=self._messages,
            tools=self._tool_defs,
        )
        self._messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            text = "".join(block.text for block in response.content if block.type == "text")
            return ChatTurn(text=text, tool_calls=[])

        calls = [
            ToolCall(id=block.id, name=block.name, input=block.input)
            for block in response.content
            if block.type == "tool_use"
        ]
        return ChatTurn(text=None, tool_calls=calls)

    def record_tool_results(self, results: list[tuple[ToolCall, str]]) -> None:
        """Append one `tool_result` content block per executed call, as Anthropic expects."""
        content = [
            {"type": "tool_result", "tool_use_id": call.id, "content": result_text}
            for call, result_text in results
        ]
        self._messages.append({"role": "user", "content": content})


class GeminiChatAdapter:
    """Drives a tool-use conversation against the Gemini API (`google-genai`)."""

    def __init__(
        self,
        client: Any,  # google.genai.Client
        model: str,
        system: str,
        user_message: str,
        tools: list[ToolSpec],
        max_tokens: int = 2048,
    ):
        from google.genai import types

        self._types = types
        self._client = client
        self._model = model
        self._config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool.name,
                            description=tool.description,
                            parameters_json_schema=tool.input_schema,
                        )
                        for tool in tools
                    ]
                )
            ],
        )
        self._contents: list[Any] = [types.Content(role="user", parts=[types.Part(text=user_message)])]

    async def send(self) -> ChatTurn:
        """Call `generate_content` with the history built so far, and normalize the reply."""
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=self._contents,
            config=self._config,
        )
        # Append the model's turn verbatim, in its own native Content shape,
        # so the next call sees exactly what it said (including any
        # function_call parts) - mirrors how the Anthropic adapter replays
        # `response.content` back into `self._messages`.
        self._contents.append(response.candidates[0].content)

        function_calls = response.function_calls or []
        if not function_calls:
            return ChatTurn(text=response.text or "", tool_calls=[])

        calls = [
            ToolCall(id=fc.id or fc.name, name=fc.name, input=dict(fc.args or {}))
            for fc in function_calls
        ]
        return ChatTurn(text=None, tool_calls=calls)

    def record_tool_results(self, results: list[tuple[ToolCall, str]]) -> None:
        """Append one `function_response` part per executed call, as Gemini expects."""
        parts = [
            self._types.Part(
                function_response=self._types.FunctionResponse(
                    id=call.id, name=call.name, response={"output": result_text}
                )
            )
            for call, result_text in results
        ]
        self._contents.append(self._types.Content(role="user", parts=parts))
