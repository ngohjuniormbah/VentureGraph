"""
Shared Instructor + Anthropic client used by every LLM-backed agent in
VentureGraph (the ORKG agent and the benchmark-extraction agent).

Why Instructor: both agents need the LLM's output to be a specific Pydantic
model, every time, or fail loudly rather than silently return malformed
JSON. Instructor wraps the Anthropic client so that a Pydantic model can be
passed as `response_model=...`; it turns that model into a tool-call schema,
validates Claude's response against it, and automatically reprompts Claude
with the validation error if the first attempt doesn't parse. That's a much
stronger guarantee than parsing free-text JSON out of a chat response by
hand, which is why we use it instead of hand-rolled prompt parsing or a
general-purpose framework like LangChain (Instructor does one thing -
structured, validated output - and does it with less indirection).

On the model version: the task brief mentions Claude 3.5 Sonnet, but per
current guidance we default to the newest Sonnet model instead of pinning
to that older snapshot, since 3.5 Sonnet has since been superseded. The
model id is centralized here and overridable via the `VENTUREGRAPH_MODEL`
environment variable (or a direct function argument) if you need to pin a
specific snapshot for reproducibility.

This module also exposes async constructors (`get_async_instructor_client`,
`get_async_anthropic_client`), used by the Venture Catalyst
(`src.agents.venture_catalyst`), which needs to run several independent LLM
calls concurrently (ideation, competitor research, and TAM calculation for
three ideas at once) rather than one at a time.
"""

import os

import instructor
from anthropic import Anthropic, AsyncAnthropic

DEFAULT_MODEL = os.environ.get("VENTUREGRAPH_MODEL", "claude-sonnet-5")


def get_instructor_client() -> instructor.Instructor:
    """
    Build a synchronous Instructor-wrapped Anthropic client.

    Data flow:
        Reads the Anthropic API key from the `ANTHROPIC_API_KEY`
        environment variable (via the standard `anthropic.Anthropic()`
        constructor) and wraps that client with `instructor.from_anthropic`,
        so callers can pass `response_model=<PydanticModel>` to
        `client.messages.create(...)` and get back a validated instance of
        that model instead of a raw API response.

    Returns:
        An `instructor.Instructor` client, drop-in compatible with the
        Anthropic client's `.messages.create()` interface plus the added
        `response_model` parameter.

    Raises:
        anthropic.AnthropicError: If `ANTHROPIC_API_KEY` is not set.
    """
    return instructor.from_anthropic(Anthropic())


def get_async_instructor_client() -> instructor.Instructor:
    """
    Build an async Instructor-wrapped Anthropic client.

    Data flow:
        Same as `get_instructor_client`, but wraps `anthropic.AsyncAnthropic`
        instead, so `await client.messages.create(response_model=...)` can
        be run concurrently (via `asyncio.gather`) with other LLM calls -
        used throughout `src.agents.venture_catalyst` so ideation,
        competitor research, and TAM calculation for multiple startup ideas
        can happen in parallel instead of one full round trip at a time.

    Returns:
        An async `instructor.Instructor` client.
    """
    return instructor.from_anthropic(AsyncAnthropic())


def get_async_anthropic_client() -> AsyncAnthropic:
    """
    Build a plain (non-Instructor) async Anthropic client.

    Data flow:
        Used by `src.agents.tool_loop.run_agentic_tool_loop`, which needs
        raw access to the `tools`/`tool_use` message format that
        Instructor's `response_model` abstraction doesn't expose - the
        tool loop is a different pattern (multi-turn, model-decides-when-
        to-call-a-tool) from Instructor's single-shot structured
        extraction.

    Returns:
        A plain `anthropic.AsyncAnthropic` client.
    """
    return AsyncAnthropic()
