"""
Shared LLM client construction for every agent in VentureGraph (the ORKG
agent, the benchmark-extraction agent, and the Venture Catalyst).

Provider switching (`LLM_PROVIDER`)
--------------------------------------
Every agent in this codebase is written against two provider-agnostic
seams, not against the Anthropic SDK directly:

1. Structured single-shot extraction goes through **Instructor**
   (`get_instructor_client` / `get_async_instructor_client`), which exposes
   the same `client.messages.create(..., response_model=SomeModel)` call
   shape regardless of which provider is behind it - Instructor's Anthropic
   and Gemini (`google-genai`) integrations both accept `system`, `messages`,
   and `response_model` and return a validated Pydantic instance. This is
   why `src/agents/orkg_agent.py`, `benchmark_agent.py`, and the ideation/
   structuring calls in `venture_catalyst.py` never mention a provider by
   name - they call whatever client this module hands them.
2. Multi-turn agentic tool use goes through the `ChatAdapter` protocol
   (`src/agents/tool_loop.py`), implemented per provider in
   `src/agents/chat_adapters.py`. `get_chat_adapter` below picks the right
   one.

Both factories read `LLM_PROVIDER` (`"anthropic"` or `"gemini"`, default
`"anthropic"`) and dispatch accordingly - so switching providers is an
environment variable, not a code change. This exists because Claude's API
is pay-as-you-go with no ongoing free tier, while Google's Gemini API
(via Google AI Studio) has a real free tier - useful for testing this
project without spending money before committing to Claude for production.
Set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY` to test for free; unset (or
`LLM_PROVIDER=anthropic`) with `ANTHROPIC_API_KEY` for production.

Why Instructor over LangChain in the first place: both agents need the
LLM's output to be a specific Pydantic model, every time, or fail loudly
rather than silently return malformed JSON. Instructor turns a Pydantic
model into a tool-call schema, validates the response against it, and
automatically reprompts on a validation failure - a stronger guarantee
than hand-rolled JSON parsing, and a much smaller dependency than a
general-purpose framework like LangChain for a need this narrow.

On the model version: the original task brief mentioned Claude 3.5 Sonnet;
`DEFAULT_MODEL` instead defaults to the current Sonnet model for the
Anthropic provider (3.5 Sonnet has since been superseded), and to a current
fast/capable Gemini model for the Gemini provider. Both are overridable via
the `VENTUREGRAPH_MODEL` environment variable if you need to pin a specific
snapshot for reproducibility.
"""

import os

import instructor
from anthropic import Anthropic, AsyncAnthropic

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").lower()

_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-2.5-flash",
}

if LLM_PROVIDER not in _DEFAULT_MODELS:
    raise ValueError(
        f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r} (expected 'anthropic' or 'gemini')"
    )

DEFAULT_MODEL = os.environ.get("VENTUREGRAPH_MODEL") or _DEFAULT_MODELS[LLM_PROVIDER]


def get_instructor_client() -> instructor.Instructor:
    """
    Build a synchronous Instructor client for the configured provider.

    Data flow:
        Reads `LLM_PROVIDER` and builds the matching native client -
        `anthropic.Anthropic()` (reads `ANTHROPIC_API_KEY`) or
        `google.genai.Client(api_key=...)` (reads `GEMINI_API_KEY`) - then
        wraps it with Instructor so callers can pass
        `response_model=<PydanticModel>` to `client.messages.create(...)`
        and get back a validated instance of that model instead of a raw
        API response, regardless of provider.

    Returns:
        An `instructor.Instructor` client exposing `.messages.create(...)`.

    Raises:
        KeyError: If the provider's required API key env var isn't set.
    """
    if LLM_PROVIDER == "gemini":
        from google import genai

        return instructor.from_genai(genai.Client(api_key=os.environ["GEMINI_API_KEY"]))
    return instructor.from_anthropic(Anthropic())


def get_async_instructor_client() -> instructor.Instructor:
    """
    Build an async Instructor client for the configured provider.

    Data flow:
        Same as `get_instructor_client`, but wraps the async native client
        (`anthropic.AsyncAnthropic` or `google.genai.Client` in async mode)
        so `await client.messages.create(response_model=...)` can be run
        concurrently (via `asyncio.gather`) with other LLM calls - used
        throughout `src.agents.venture_catalyst` so ideation, competitor
        research, and TAM calculation for multiple startup ideas can
        happen in parallel instead of one full round trip at a time.

    Returns:
        An async `instructor.Instructor` client exposing `.messages.create(...)`.

    Raises:
        KeyError: If the provider's required API key env var isn't set.
    """
    if LLM_PROVIDER == "gemini":
        from google import genai

        return instructor.from_genai(genai.Client(api_key=os.environ["GEMINI_API_KEY"]), use_async=True)
    return instructor.from_anthropic(AsyncAnthropic())


def get_chat_adapter(
    system: str,
    user_message: str,
    tools: list,
    model: str | None = None,
    max_tokens: int = 2048,
):
    """
    Build a `ChatAdapter` (see `src.agents.tool_loop`) for the configured provider.

    Data flow:
        Used by `src.agents.venture_catalyst.research_competitors` and
        `calculate_tam`, which need Claude/Gemini to autonomously call
        tools across multiple turns - a pattern Instructor's single-shot
        `response_model` doesn't cover. Reads `LLM_PROVIDER` and builds the
        matching native client plus its `ChatAdapter` implementation
        (`src.agents.chat_adapters.AnthropicChatAdapter` or
        `GeminiChatAdapter`), already primed with `system`, `user_message`,
        and `tools`. The returned adapter is handed to
        `src.agents.tool_loop.run_agentic_tool_loop`, which drives it
        without knowing which provider is underneath.

    Args:
        system: System prompt for the conversation.
        user_message: The initial user turn.
        tools: The `ToolSpec`s the model may call.
        model: Model id to use; defaults to `DEFAULT_MODEL`.
        max_tokens: Max tokens per model response.

    Returns:
        A `ChatAdapter` ready for `run_agentic_tool_loop`.

    Raises:
        KeyError: If the provider's required API key env var isn't set.
    """
    from src.agents.chat_adapters import AnthropicChatAdapter, GeminiChatAdapter

    model = model or DEFAULT_MODEL

    if LLM_PROVIDER == "gemini":
        from google import genai

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        return GeminiChatAdapter(client, model, system, user_message, tools, max_tokens)

    return AnthropicChatAdapter(AsyncAnthropic(), model, system, user_message, tools, max_tokens)
