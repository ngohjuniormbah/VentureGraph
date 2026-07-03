"""
Investment Thesis report renderer.

Deterministic Markdown assembly - no LLM call - that combines everything
the three VentureGraph layers produced into a single document: the paper's
scientific summary (parser + extractor layers), its ORKG contribution table
(Scientific Intelligence layer), and the Venture Catalyst's startup ideas
with their competitor landscapes and TAM calculations (this layer). Kept
LLM-free for the same reason `src.comparison.comparison_engine` is: once
the underlying facts are already structured Pydantic objects, formatting
them into Markdown is mechanical and should be reproducible.
"""

from src.schemas.orkg import ORKGContribution
from src.schemas.scientific_essence import ScientificEssence
from src.schemas.venture import IdeaDossier


def _render_scientific_summary(essence: ScientificEssence) -> str:
    """Render the paper's ScientificEssence as a Markdown section."""
    authors = ", ".join(author.name for author in essence.authors) or "Unknown"
    return (
        "## Scientific Summary\n\n"
        f"**Title:** {essence.paper_title}\n\n"
        f"**Authors:** {authors}\n\n"
        f"**Problem Statement:** {essence.problem_statement}\n\n"
        f"**Methodology:** {essence.methodology}\n\n"
        f"**Key Results:** {essence.key_results}"
    )


def _render_orkg_table(contribution: ORKGContribution) -> str:
    """Render an ORKGContribution's triples as a Markdown Subject/Predicate/Object table."""
    lines = [
        "## ORKG Contribution",
        "",
        f"**Research Problem:** {contribution.research_problem}",
        "",
        f"**Contribution:** {contribution.contribution_label}",
        "",
        "| Subject | Predicate | Object |",
        "|---|---|---|",
    ]
    for triple in contribution.triples:
        lines.append(f"| {triple.subject} | {triple.predicate} | {triple.object} |")
    return "\n".join(lines)


def _render_idea_dossier(dossier: IdeaDossier, index: int) -> str:
    """Render one IdeaDossier (idea + competitors + TAM) as a Markdown subsection."""
    idea = dossier.idea
    competitors = dossier.competitors
    tam = dossier.tam

    lines = [
        f"### {index}. {idea.name}",
        "",
        f"**Description:** {idea.description}",
        "",
        f"**Target Customer:** {idea.target_customer}",
        "",
        f"**Commercial Angle:** {idea.commercial_angle}",
        "",
        f'> Grounded in methodology: "{idea.methodology_basis}"',
        "",
        "**Competitive Landscape**",
        "",
        competitors.summary,
        "",
        "| Competitor | Description | URL |",
        "|---|---|---|",
    ]
    for competitor in competitors.competitors:
        lines.append(f"| {competitor.name} | {competitor.description} | {competitor.url or '—'} |")

    lines += [
        "",
        "**Total Addressable Market (TAM)**",
        "",
        f"**Estimated TAM: ${tam.tam_usd:,.0f}**",
        "",
        "Assumptions:",
    ]
    lines += [f"- {assumption}" for assumption in tam.assumptions] or ["- (none stated)"]

    lines += [
        "",
        "Calculation:",
        "```python",
        tam.calculation_code,
        "```",
        "",
        "Output:",
        "```",
        tam.calculation_output,
        "```",
        "",
        "Data sources: " + (", ".join(tam.data_sources) if tam.data_sources else "—"),
    ]
    return "\n".join(lines)


def render_investment_thesis_markdown(
    essence: ScientificEssence,
    orkg_contribution: ORKGContribution,
    idea_dossiers: list[IdeaDossier],
) -> str:
    """
    Assemble the final Investment Thesis report as a single Markdown string.

    Data flow:
        Takes the outputs of all three VentureGraph layers - the paper's
        `ScientificEssence`, its `ORKGContribution`, and the Venture
        Catalyst's list of `IdeaDossier`s - and renders them into one
        document: a scientific summary section, an ORKG Subject/Predicate/
        Object table, and one subsection per startup idea (its competitive
        landscape table and its TAM calculation, code included). `main.py`'s
        `thesis` command writes the result to `<pdf-stem>.thesis.md`.

    Args:
        essence: The paper's `ScientificEssence`.
        orkg_contribution: The paper's `ORKGContribution`.
        idea_dossiers: The Venture Catalyst's `IdeaDossier`s (typically
            three, one per startup idea).

    Returns:
        The complete Investment Thesis as a Markdown string.
    """
    parts = [
        f"# Investment Thesis: {essence.paper_title}",
        _render_scientific_summary(essence),
        _render_orkg_table(orkg_contribution),
        "## Startup Ideas & Market Analysis",
    ]
    parts.extend(_render_idea_dossier(dossier, index) for index, dossier in enumerate(idea_dossiers, start=1))
    return "\n\n".join(parts)
