"""
Web search tool integration for the Venture Catalyst.

The "outdated training data" problem, and how this solves it
----------------------------------------------------------------
Claude's parametric knowledge has a training cutoff, but competitor
landscapes and market-size figures are exactly the kind of fact that goes
stale continuously - a competitor list from a training snapshot could be a
year or more out of date, and the model has no way to know that on its own.
Asking Claude to just "list competitors" or "estimate TAM" from memory
would silently launder stale (or invented) numbers as if they were current.

The fix used throughout the Venture Catalyst (`src/agents/venture_catalyst.py`)
is **tool use**: Claude is given a `web_search` tool it can call during an
agentic loop (see `src/agents/tool_loop.py`). Each call here hits a live
search API - real HTTP requests made at request time, not anything baked
into the model - and the actual retrieved titles/URLs/snippets are fed back
into the conversation as the tool's result. Claude's final answer is then
synthesized from that fresh retrieved text, and every downstream schema
(`CompetitorLandscape`, `TAMEstimate`) keeps the source URLs that were
actually used, so a human can click through and verify the numbers rather
than trusting the model's say-so.

Provider choice (Tavily vs. Brave) is a runtime setting, not a code fork:
both are simple REST search endpoints behind the same `SearchProvider`
interface, selected via the `SEARCH_PROVIDER` environment variable.
"""

import os
from typing import Protocol

import httpx
from pydantic import BaseModel


class SearchResult(BaseModel):
    """One web search result, in a provider-agnostic shape."""

    title: str
    url: str
    snippet: str


class SearchProvider(Protocol):
    """Common interface both search backends implement."""

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        ...


class TavilySearchProvider:
    """
    Search via Tavily (https://tavily.com) - a search API purpose-built for
    LLM agents, returning clean, pre-summarized snippets rather than raw
    HTML search-result pages.
    """

    def __init__(self, api_key: str | None = None):
        # Imported lazily so `tavily-python` is only required when this
        # provider is actually selected.
        from tavily import AsyncTavilyClient

        self._client = AsyncTavilyClient(api_key=api_key or os.environ["TAVILY_API_KEY"])

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """
        Run a live Tavily search and return normalized results.

        Data flow:
            Sends `query` to the Tavily API over HTTPS at call time,
            receives a JSON payload with a `results` list, and maps each
            entry's `title`/`url`/`content` into a `SearchResult`. This is
            the point where "current, real" information enters the system.
        """
        response = await self._client.search(query, max_results=max_results)
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
            )
            for item in response.get("results", [])
        ]


class BraveSearchProvider:
    """Search via the Brave Search API (https://brave.com/search/api)."""

    _ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ["BRAVE_API_KEY"]

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """
        Run a live Brave Search query and return normalized results.

        Data flow:
            Issues a GET request to Brave's REST API at call time,
            authenticated with `BRAVE_API_KEY`, and maps the
            `web.results[].title/url/description` fields into
            `SearchResult`s.
        """
        async with httpx.AsyncClient(timeout=15.0) as http_client:
            response = await http_client.get(
                self._ENDPOINT,
                params={"q": query, "count": max_results},
                headers={"Accept": "application/json", "X-Subscription-Token": self._api_key},
            )
            response.raise_for_status()
            payload = response.json()

        results = payload.get("web", {}).get("results", [])[:max_results]
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
            )
            for item in results
        ]


def get_search_provider() -> SearchProvider:
    """
    Build the search provider configured via the `SEARCH_PROVIDER` env var.

    Data flow:
        Reads `SEARCH_PROVIDER` (defaulting to `"tavily"`), and constructs
        the matching provider class, which will read its own API key from
        the environment (`TAVILY_API_KEY` or `BRAVE_API_KEY`) when it's
        actually used. Called once per `web_search` tool registration in
        `src.agents.venture_catalyst`.

    Returns:
        A `SearchProvider` instance (`TavilySearchProvider` or
        `BraveSearchProvider`).

    Raises:
        ValueError: If `SEARCH_PROVIDER` is set to something else.
    """
    provider_name = os.environ.get("SEARCH_PROVIDER", "tavily").lower()
    if provider_name == "tavily":
        return TavilySearchProvider()
    if provider_name == "brave":
        return BraveSearchProvider()
    raise ValueError(f"Unknown SEARCH_PROVIDER: {provider_name!r} (expected 'tavily' or 'brave')")


def format_search_results_for_llm(results: list[SearchResult]) -> str:
    """
    Render search results as plain text for a tool_result message.

    Data flow:
        Takes the `SearchResult`s returned by a `SearchProvider.search()`
        call and formats them as a numbered list (title, URL, snippet) that
        gets sent back to Claude as the `web_search` tool's result inside
        `src.agents.tool_loop.run_agentic_tool_loop`.

    Args:
        results: Search results to render.

    Returns:
        A plain-text block, or `"No results found."` if `results` is empty.
    """
    if not results:
        return "No results found."
    return "\n\n".join(
        f"[{i}] {result.title}\nURL: {result.url}\n{result.snippet}"
        for i, result in enumerate(results, start=1)
    )
