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
"""

import os

import instructor
from anthropic import Anthropic

DEFAULT_MODEL = os.environ.get("VENTUREGRAPH_MODEL", "claude-sonnet-5")


def get_instructor_client() -> instructor.Instructor:
    """
    Build an Instructor-wrapped Anthropic client for structured LLM calls.

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
