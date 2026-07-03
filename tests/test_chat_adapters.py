"""
Tests for the provider-specific ChatAdapter implementations.

These fake each provider's native client (Anthropic's `messages.create`,
Gemini's `client.aio.models.generate_content`) at the shape level, since
`AnthropicChatAdapter`/`GeminiChatAdapter` are exactly the code that
translates between that shape and the provider-agnostic `ChatTurn`/
`ToolCall` the rest of the pipeline (`tool_loop.py`, `venture_catalyst.py`)
relies on. No real API key or network access is used or required.
"""

import types

from src.agents.chat_adapters import AnthropicChatAdapter, GeminiChatAdapter
from src.agents.tool_loop import ToolCall, ToolSpec

SIMPLE_TOOL = ToolSpec(
    name="web_search",
    description="search the web",
    input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    executor=lambda query: query,  # not exercised - adapters don't call executors themselves
)


# --- AnthropicChatAdapter -----------------------------------------------------


def _anthropic_text_block(text: str):
    return types.SimpleNamespace(type="text", text=text)


def _anthropic_tool_use_block(tool_use_id: str, name: str, tool_input: dict):
    return types.SimpleNamespace(type="tool_use", id=tool_use_id, name=name, input=tool_input)


def _anthropic_response(stop_reason: str, content: list):
    return types.SimpleNamespace(stop_reason=stop_reason, content=content)


class _FakeAnthropicMessages:
    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        # Snapshot mutable args (e.g. `messages`) at call time - the adapter
        # keeps appending to the same list object after this call returns,
        # so a shallow reference would silently "see" later appends too.
        self.calls.append({key: (list(value) if isinstance(value, list) else value) for key, value in kwargs.items()})
        return self._responses.pop(0)


class _FakeAnthropicClient:
    def __init__(self, responses: list):
        self.messages = _FakeAnthropicMessages(responses)


async def test_anthropic_adapter_normalizes_tool_use_turn():
    fake_client = _FakeAnthropicClient(
        responses=[_anthropic_response("tool_use", [_anthropic_tool_use_block("call_1", "web_search", {"query": "q"})])]
    )
    adapter = AnthropicChatAdapter(fake_client, "fake-model", "system prompt", "find competitors", [SIMPLE_TOOL])

    turn = await adapter.send()

    assert turn.text is None
    assert turn.tool_calls == [ToolCall(id="call_1", name="web_search", input={"query": "q"})]
    # The tool definition sent to the API matches SIMPLE_TOOL's schema.
    sent_tools = fake_client.messages.calls[0]["tools"]
    assert sent_tools[0]["name"] == "web_search"
    assert sent_tools[0]["input_schema"] == SIMPLE_TOOL.input_schema


async def test_anthropic_adapter_normalizes_final_text_turn():
    fake_client = _FakeAnthropicClient(responses=[_anthropic_response("end_turn", [_anthropic_text_block("done")])])
    adapter = AnthropicChatAdapter(fake_client, "fake-model", "system prompt", "find competitors", [SIMPLE_TOOL])

    turn = await adapter.send()

    assert turn.text == "done"
    assert turn.tool_calls == []


async def test_anthropic_adapter_records_tool_results_as_tool_result_blocks():
    fake_client = _FakeAnthropicClient(
        responses=[
            _anthropic_response("tool_use", [_anthropic_tool_use_block("call_1", "web_search", {"query": "q"})]),
            _anthropic_response("end_turn", [_anthropic_text_block("done")]),
        ]
    )
    adapter = AnthropicChatAdapter(fake_client, "fake-model", "system prompt", "find competitors", [SIMPLE_TOOL])

    turn = await adapter.send()
    adapter.record_tool_results([(turn.tool_calls[0], "Acme Corp is a competitor.")])
    await adapter.send()

    second_call_messages = fake_client.messages.calls[1]["messages"]
    tool_result_message = second_call_messages[-1]
    assert tool_result_message["role"] == "user"
    assert tool_result_message["content"] == [
        {"type": "tool_result", "tool_use_id": "call_1", "content": "Acme Corp is a competitor."}
    ]


# --- GeminiChatAdapter ---------------------------------------------------------


class _FakeGeminiModels:
    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def generate_content(self, **kwargs):
        # Snapshot mutable args (e.g. `contents`) at call time - see the
        # matching comment on _FakeAnthropicMessages.create above.
        self.calls.append({key: (list(value) if isinstance(value, list) else value) for key, value in kwargs.items()})
        return self._responses.pop(0)


class _FakeGeminiAio:
    def __init__(self, responses: list):
        self.models = _FakeGeminiModels(responses)


class _FakeGeminiClient:
    def __init__(self, responses: list):
        self.aio = _FakeGeminiAio(responses)


def _gemini_response(function_calls, text, model_content):
    """Build a fake google.genai GenerateContentResponse-shaped object."""
    return types.SimpleNamespace(
        function_calls=function_calls or None,
        text=text,
        candidates=[types.SimpleNamespace(content=model_content)],
    )


async def test_gemini_adapter_normalizes_function_call_turn():
    from google.genai import types as genai_types

    fake_function_call = genai_types.FunctionCall(id="call_1", name="web_search", args={"query": "q"})
    model_content = genai_types.Content(role="model", parts=[genai_types.Part(function_call=fake_function_call)])
    fake_client = _FakeGeminiClient(responses=[_gemini_response([fake_function_call], None, model_content)])

    adapter = GeminiChatAdapter(fake_client, "fake-model", "system prompt", "find competitors", [SIMPLE_TOOL])
    turn = await adapter.send()

    assert turn.text is None
    assert turn.tool_calls == [ToolCall(id="call_1", name="web_search", input={"query": "q"})]

    # The tool definition sent to the API matches SIMPLE_TOOL's schema.
    sent_config = fake_client.aio.models.calls[0]["config"]
    function_declaration = sent_config.tools[0].function_declarations[0]
    assert function_declaration.name == "web_search"
    assert function_declaration.parameters_json_schema == SIMPLE_TOOL.input_schema


async def test_gemini_adapter_normalizes_final_text_turn():
    from google.genai import types as genai_types

    model_content = genai_types.Content(role="model", parts=[genai_types.Part(text="done")])
    fake_client = _FakeGeminiClient(responses=[_gemini_response([], "done", model_content)])

    adapter = GeminiChatAdapter(fake_client, "fake-model", "system prompt", "find competitors", [SIMPLE_TOOL])
    turn = await adapter.send()

    assert turn.text == "done"
    assert turn.tool_calls == []


async def test_gemini_adapter_records_tool_results_as_function_response_parts():
    from google.genai import types as genai_types

    fake_function_call = genai_types.FunctionCall(id="call_1", name="web_search", args={"query": "q"})
    model_content = genai_types.Content(role="model", parts=[genai_types.Part(function_call=fake_function_call)])
    fake_client = _FakeGeminiClient(
        responses=[
            _gemini_response([fake_function_call], None, model_content),
            _gemini_response([], "done", genai_types.Content(role="model", parts=[genai_types.Part(text="done")])),
        ]
    )

    adapter = GeminiChatAdapter(fake_client, "fake-model", "system prompt", "find competitors", [SIMPLE_TOOL])
    turn = await adapter.send()
    adapter.record_tool_results([(turn.tool_calls[0], "Acme Corp is a competitor.")])
    await adapter.send()

    second_call_contents = fake_client.aio.models.calls[1]["contents"]
    tool_result_content = second_call_contents[-1]
    assert tool_result_content.role == "user"
    function_response = tool_result_content.parts[0].function_response
    assert function_response.name == "web_search"
    assert function_response.response == {"output": "Acme Corp is a competitor."}
