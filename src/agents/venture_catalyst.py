"""
The Venture Catalyst: turns a paper's methodology into an Investment Thesis.

Pipeline (see `run_venture_catalyst` for the orchestration):
    1. Startup Ideator (single-shot, Instructor): reads
       `ScientificEssence.methodology` and proposes exactly three
       commercial use cases, each grounded in a quoted piece of the
       methodology text (`generate_startup_ideas`).
    2. Market Intelligence (agentic tool loop, per idea): gives Claude a
       `web_search` tool (see `src.tools.web_search`) and asks it to find
       current competitors; the loop runs until Claude has gathered enough
       live search results to answer, then a short follow-up Instructor
       call structures that free-text answer into a `CompetitorLandscape`
       (`research_competitors`).
    3. TAM Calculator (agentic tool loop, per idea): gives Claude both the
       `web_search` tool (to find market-size figures/industry reports)
       and an `execute_python` tool (see `src.tools.code_executor`) to
       compute the TAM from whatever numbers it finds, then structures the
       result into a `TAMEstimate` that keeps the executed code and its
       output for auditability (`calculate_tam`).
    4. Steps 2 and 3 run concurrently for each idea, and all three ideas
       run concurrently with each other (`asyncio.gather` in
       `build_idea_dossier` and `run_venture_catalyst`) - see the "why
       async" note below.

Why this module is fully async
---------------------------------
A single idea's dossier needs two independent multi-turn agentic
conversations (competitor research, TAM calculation), and there are three
ideas. Run sequentially, that's up to six full tool-use conversations - each
involving several Claude round trips plus live search/code-execution calls
- stacked one after another. None of the six branches read from each other,
so there's no correctness reason to serialize them. `asyncio.gather` runs
them concurrently instead, so the wall-clock cost of this whole layer is
roughly the cost of the *slowest single branch*, not the sum of all of
them - the difference between, say, 20 seconds and 2 minutes.
"""

import asyncio

from anthropic import AsyncAnthropic

from src.agents.llm_client import DEFAULT_MODEL, get_async_anthropic_client, get_async_instructor_client
from src.agents.tool_loop import ToolSpec, run_agentic_tool_loop
from src.schemas.scientific_essence import ScientificEssence
from src.schemas.venture import CompetitorLandscape, IdeaDossier, StartupIdea, StartupIdeas, TAMEstimate
from src.tools.code_executor import execute_python
from src.tools.web_search import format_search_results_for_llm, get_search_provider

IDEATOR_SYSTEM_PROMPT = """\
You are a deep-tech venture analyst. Your job is to read a scientific \
paper's methodology and identify commercially viable use cases that a \
startup could be built on.

Strict rules:
- Ground every idea in something the methodology actually does - quote the \
exact supporting text from the methodology in `methodology_basis` for each \
idea, verbatim.
- Propose exactly three ideas, and make them meaningfully distinct from \
each other (different customers, different products, or different value \
propositions), not three phrasings of the same idea.
- Write your reasoning before naming each idea: what capability does the \
methodology provide, and who would pay for that capability?
- Do not invent capabilities the methodology doesn't describe.
"""

MARKET_INTEL_SYSTEM_PROMPT = """\
You are a market intelligence analyst. You have a `web_search` tool - use \
it to find CURRENT, real competitors for the startup idea you're given. \
Your own knowledge may be outdated, so always search rather than relying \
on what you already know; issue multiple searches with different phrasings \
if the first one doesn't turn up clear competitors. When you have enough \
information, give a final plain-text answer that: names each real \
competitor you found (from the search results, not from memory), briefly \
describes what it does, gives its URL, and cites which search result URLs \
you used. If you can't find clear competitors after a couple of searches, \
say so honestly instead of inventing some.
"""

TAM_SYSTEM_PROMPT = """\
You are a market-sizing analyst. You have two tools: `web_search` (to find \
current industry-report figures - market size, growth rate, number of \
potential customers/units) and `execute_python` (to compute the Total \
Addressable Market from numbers you've found). Your own knowledge of \
market-size figures may be outdated, so always search for current figures \
rather than recalling them from memory. Do the arithmetic by writing a \
short Python script and running it with `execute_python` - do not compute \
the final number by hand. When you have a result, give a final plain-text \
answer that states the TAM in USD, lists your assumptions, includes the \
exact Python code you ran and its output, and cites the URLs the input \
figures came from.
"""

_STRUCTURING_SYSTEM_PROMPT = (
    "Reformat the text you're given into the requested schema. Do not add, "
    "infer, or invent any fact, number, competitor, or URL that isn't "
    "already present in the text - only restructure what's there."
)


def _web_search_tool_spec(max_results: int = 5) -> ToolSpec:
    """
    Build the `web_search` ToolSpec.

    The search provider (and the API key / optional dependency it needs)
    is constructed lazily, inside `executor`, rather than here - so simply
    building this spec (which happens on every `research_competitors`/
    `calculate_tam` call, whether or not the model ends up calling the
    tool) never requires `TAVILY_API_KEY`/`BRAVE_API_KEY` to be set or
    `tavily-python` to be installed. That cost is only paid the moment the
    model actually decides to search.
    """

    async def executor(query: str) -> str:
        provider = get_search_provider()
        results = await provider.search(query, max_results=max_results)
        return format_search_results_for_llm(results)

    return ToolSpec(
        name="web_search",
        description=(
            "Search the live web for current information. Use this for anything "
            "that could have changed since your training data - competitor names, "
            "company details, market size figures, industry reports. Returns a "
            "numbered list of results, each with a title, URL, and snippet."
        ),
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query to run."}},
            "required": ["query"],
        },
        executor=executor,
    )


def _execute_python_tool_spec() -> ToolSpec:
    """Build the `execute_python` ToolSpec, bound to the sandboxed executor."""

    async def executor(code: str) -> str:
        return await execute_python(code)

    return ToolSpec(
        name="execute_python",
        description=(
            "Execute a short, self-contained Python script and return its stdout. "
            "Use this to compute the Total Addressable Market from numbers you've "
            "gathered via web_search (e.g. multiplying a market-size figure by a "
            "penetration rate). The script must print() its result."
        ),
        input_schema={
            "type": "object",
            "properties": {"code": {"type": "string", "description": "The Python script to execute."}},
            "required": ["code"],
        },
        executor=executor,
    )


async def generate_startup_ideas(
    essence: ScientificEssence,
    model: str = DEFAULT_MODEL,
    client=None,
) -> StartupIdeas:
    """
    Analyze a paper's methodology and propose exactly three commercial use cases.

    Data flow:
        1. Takes the paper's `ScientificEssence` (specifically
           `.methodology`, plus `.problem_statement`/`.key_results` as
           supporting context).
        2. Sends it to Claude via async Instructor with
           `response_model=StartupIdeas`, which enforces (and reprompts
           until satisfied) that exactly three ideas are returned, each
           with a verbatim `methodology_basis` quote.
        3. Returns the resulting `StartupIdeas`, which
           `run_venture_catalyst` fans out into three concurrent
           `build_idea_dossier` calls.

    Args:
        essence: The paper's `ScientificEssence`.
        model: Anthropic model id to use.
        client: Optional pre-built async Instructor client (for testing).

    Returns:
        A `StartupIdeas` with exactly three `StartupIdea`s.
    """
    client = client or get_async_instructor_client()

    user_prompt = f"""\
Paper title: {essence.paper_title}

Problem statement: {essence.problem_statement}

Methodology:
{essence.methodology}

Key results: {essence.key_results}

Identify exactly three distinct commercial use cases for a startup built on \
this methodology.
"""

    return await client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0.4,
        system=IDEATOR_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        response_model=StartupIdeas,
    )


async def research_competitors(
    idea: StartupIdea,
    model: str = DEFAULT_MODEL,
    tool_client: AsyncAnthropic | None = None,
    structuring_client=None,
) -> CompetitorLandscape:
    """
    Find current competitors for a startup idea via live web search.

    Data flow:
        1. Builds a `web_search` `ToolSpec` and hands it to
           `run_agentic_tool_loop` along with a description of `idea`,
           letting Claude decide how many searches to run and with what
           queries.
        2. The loop returns Claude's final free-text synthesis of what it
           found (grounded in real, live search results - see
           `src.tools.web_search` for why that matters).
        3. A second, short Instructor call reformats that free text into a
           `CompetitorLandscape` - explicitly instructed not to add any
           fact beyond what's already in the text, so this structuring
           step can't reintroduce hallucination after the grounded loop.

    Args:
        idea: The `StartupIdea` to research competitors for.
        model: Anthropic model id to use.
        tool_client: Optional pre-built async Anthropic client (for
            testing); defaults to `get_async_anthropic_client()`.
        structuring_client: Optional pre-built async Instructor client (for
            testing); defaults to `get_async_instructor_client()`.

    Returns:
        A `CompetitorLandscape` for this idea.
    """
    tool_client = tool_client or get_async_anthropic_client()
    structuring_client = structuring_client or get_async_instructor_client()

    user_message = f"""\
Startup idea: {idea.name}
Description: {idea.description}
Target customer: {idea.target_customer}

Use web_search to find current, real competitors for this idea. Search with \
multiple relevant queries if needed (e.g. the product category, and the \
target customer's industry). Then give your final answer as described in \
your instructions.
"""

    final_answer = await run_agentic_tool_loop(
        tool_client,
        MARKET_INTEL_SYSTEM_PROMPT,
        user_message,
        tools=[_web_search_tool_spec()],
        model=model,
    )

    return await structuring_client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=0,
        system=_STRUCTURING_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Startup idea name: {idea.name}\n\nMarket research findings:\n{final_answer}",
            }
        ],
        response_model=CompetitorLandscape,
    )


async def calculate_tam(
    idea: StartupIdea,
    model: str = DEFAULT_MODEL,
    tool_client: AsyncAnthropic | None = None,
    structuring_client=None,
) -> TAMEstimate:
    """
    Estimate a startup idea's Total Addressable Market via search + code execution.

    Data flow:
        1. Builds `web_search` and `execute_python` `ToolSpec`s and hands
           both to `run_agentic_tool_loop`, so Claude can search for
           current market-size figures and then write/run a Python script
           that computes the TAM from whatever it finds.
        2. The loop returns Claude's final free-text report (the TAM
           figure, assumptions, the executed code, its output, and
           sources).
        3. A second, short Instructor call reformats that report into a
           `TAMEstimate` - instructed not to invent any number beyond
           what's already in the text, preserving the exact executed code
           and output for auditability.

    Args:
        idea: The `StartupIdea` to estimate a TAM for.
        model: Anthropic model id to use.
        tool_client: Optional pre-built async Anthropic client (for
            testing); defaults to `get_async_anthropic_client()`.
        structuring_client: Optional pre-built async Instructor client (for
            testing); defaults to `get_async_instructor_client()`.

    Returns:
        A `TAMEstimate` for this idea.
    """
    tool_client = tool_client or get_async_anthropic_client()
    structuring_client = structuring_client or get_async_instructor_client()

    user_message = f"""\
Startup idea: {idea.name}
Description: {idea.description}
Target customer: {idea.target_customer}

Estimate the Total Addressable Market (TAM) for this idea: search for \
current industry-report figures, then write and execute a Python script \
that computes the TAM from them, then give your final answer as described \
in your instructions.
"""

    final_answer = await run_agentic_tool_loop(
        tool_client,
        TAM_SYSTEM_PROMPT,
        user_message,
        tools=[_web_search_tool_spec(), _execute_python_tool_spec()],
        model=model,
        max_turns=8,
    )

    return await structuring_client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=0,
        system=_STRUCTURING_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Startup idea name: {idea.name}\n\nTAM analysis:\n{final_answer}",
            }
        ],
        response_model=TAMEstimate,
    )


async def build_idea_dossier(idea: StartupIdea, model: str = DEFAULT_MODEL) -> IdeaDossier:
    """
    Build the full dossier (competitors + TAM) for one startup idea.

    Data flow:
        Runs `research_competitors` and `calculate_tam` concurrently via
        `asyncio.gather` - they are independent agentic tool loops that
        don't depend on each other's output - and combines the results
        with `idea` into a single `IdeaDossier`.

    Args:
        idea: The `StartupIdea` to build a dossier for.
        model: Anthropic model id to use for both sub-agents.

    Returns:
        An `IdeaDossier` combining the idea, its competitor landscape, and
        its TAM estimate.
    """
    competitors, tam = await asyncio.gather(
        research_competitors(idea, model=model),
        calculate_tam(idea, model=model),
    )
    return IdeaDossier(idea=idea, competitors=competitors, tam=tam)


async def run_venture_catalyst(essence: ScientificEssence, model: str = DEFAULT_MODEL) -> list[IdeaDossier]:
    """
    Run the full Venture Catalyst pipeline for a paper.

    Data flow:
        1. Calls `generate_startup_ideas` once to get exactly three
           `StartupIdea`s from the paper's methodology.
        2. Fans out `build_idea_dossier` across all three ideas
           concurrently via `asyncio.gather` (each of which itself runs
           two concurrent agentic tool loops - see `build_idea_dossier`),
           so all six research/calculation branches across all three
           ideas run in parallel rather than one at a time.

    Args:
        essence: The paper's `ScientificEssence`.
        model: Anthropic model id to use throughout.

    Returns:
        A list of three `IdeaDossier`s, one per startup idea.
    """
    ideas = await generate_startup_ideas(essence, model=model)
    dossiers = await asyncio.gather(*(build_idea_dossier(idea, model=model) for idea in ideas.ideas))
    return list(dossiers)
