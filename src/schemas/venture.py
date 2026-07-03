"""
Pydantic schemas for the Venture Catalyst: startup ideation, competitor
research, TAM estimation, and the final combined Investment Thesis.
"""

from pydantic import BaseModel, Field

from src.schemas.orkg import ORKGContribution
from src.schemas.scientific_essence import ScientificEssence


class StartupIdea(BaseModel):
    """
    A single commercial use case derived from a paper's methodology.

    Data flow:
        Produced by `src.agents.venture_catalyst.generate_startup_ideas`,
        one of exactly three collected into `StartupIdeas.ideas`. Each one
        is later expanded into an `IdeaDossier` (competitors + TAM).
    """

    reasoning: str = Field(
        ...,
        description=(
            "Why this is a viable commercial use case, reasoned step by "
            "step from the paper's methodology, written BEFORE the idea "
            "itself is named."
        ),
    )
    name: str = Field(..., description="A short, memorable name for the startup idea.")
    description: str = Field(..., description="A one- or two-sentence description of the product/service.")
    target_customer: str = Field(..., description="Who would buy this - the specific customer segment.")
    commercial_angle: str = Field(
        ...,
        description="What specifically from the paper's methodology makes this commercially viable or differentiated.",
    )
    methodology_basis: str = Field(
        ...,
        description=(
            "A verbatim quote from the paper's Methodology text that this "
            "idea is grounded in - must be an exact substring of the "
            "methodology text, not a paraphrase."
        ),
    )


class StartupIdeas(BaseModel):
    """
    The Startup Ideator's output: exactly three commercial use cases.

    Data flow:
        Returned by `generate_startup_ideas` from a single Instructor call
        against `essence.methodology`. `main.py`'s `thesis` command fans
        `ideas` out into three concurrent `build_idea_dossier` calls.
    """

    reasoning: str = Field(
        ...,
        description="Overall reasoning about the methodology's commercial potential, written before the ideas list.",
    )
    paper_title: str
    ideas: list[StartupIdea] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Exactly three distinct commercial use cases.",
    )


class Competitor(BaseModel):
    """One competitor found via live web search."""

    name: str
    description: str = Field(..., description="What this competitor does, based on the search results.")
    url: str | None = Field(default=None, description="URL of the source page this competitor was found on.")


class CompetitorLandscape(BaseModel):
    """
    The Market Intelligence agent's output for one startup idea.

    Data flow:
        Produced by `src.agents.venture_catalyst.research_competitors`,
        which runs an agentic `web_search` tool loop and then structures
        the loop's free-text findings into this schema. `sources` and each
        `Competitor.url` trace back to the live search results the answer
        was actually grounded in - see `src/tools/web_search.py` for why
        that matters more than trusting the model's training data.
    """

    idea_name: str
    summary: str = Field(..., description="A one- or two-sentence synthesis of the competitive landscape.")
    competitors: list[Competitor] = Field(default_factory=list)
    search_queries_used: list[str] = Field(
        default_factory=list, description="The web_search queries that were actually issued."
    )
    sources: list[str] = Field(
        default_factory=list, description="URLs of the search results the findings above were drawn from."
    )


class TAMEstimate(BaseModel):
    """
    The TAM Calculator's output for one startup idea.

    Data flow:
        Produced by `src.agents.venture_catalyst.calculate_tam`, which runs
        an agentic tool loop with both `web_search` (to find market-size
        figures) and `execute_python` (to compute the TAM from them), then
        structures the result into this schema. `calculation_code` and
        `calculation_output` are kept verbatim so the arithmetic is fully
        auditable rather than a number Claude simply asserts.
    """

    idea_name: str
    reasoning: str = Field(..., description="Explanation of the market-sizing approach taken.")
    assumptions: list[str] = Field(
        default_factory=list, description="Explicit assumptions made in the calculation (e.g. penetration rate)."
    )
    calculation_code: str = Field(..., description="The exact Python script that was executed to compute the TAM.")
    calculation_output: str = Field(..., description="The stdout produced by executing calculation_code.")
    tam_usd: float = Field(..., description="The final estimated Total Addressable Market, in US dollars.")
    data_sources: list[str] = Field(
        default_factory=list, description="URLs of the industry reports/search results the figures came from."
    )


class IdeaDossier(BaseModel):
    """One startup idea, bundled with its market research and TAM estimate."""

    idea: StartupIdea
    competitors: CompetitorLandscape
    tam: TAMEstimate


class InvestmentThesis(BaseModel):
    """
    The final combined output of the whole VentureGraph pipeline.

    Data flow:
        Assembled by `main.py`'s `thesis` command from: the paper's
        `ScientificEssence` (parser + extractor layers), its
        `ORKGContribution` (Scientific Intelligence layer), and the
        Venture Catalyst's `IdeaDossier`s (this layer). Passed to
        `src.report.investment_thesis.render_investment_thesis_markdown`
        to produce the final human-readable report.
    """

    paper_title: str
    source_file: str | None = None
    scientific_essence: ScientificEssence
    orkg_contribution: ORKGContribution
    idea_dossiers: list[IdeaDossier]
