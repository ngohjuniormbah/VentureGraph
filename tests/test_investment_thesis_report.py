"""
Tests for the deterministic Investment Thesis Markdown renderer. Builds
every input by hand (no LLM call) since rendering already-structured data
is pure formatting logic.
"""

from src.report.investment_thesis import render_investment_thesis_markdown
from src.schemas.orkg import ORKGContribution, Triple
from src.schemas.scientific_essence import Author, ScientificEssence
from src.schemas.venture import Competitor, CompetitorLandscape, IdeaDossier, StartupIdea, TAMEstimate


def _essence() -> ScientificEssence:
    return ScientificEssence(
        paper_title="Gradient Widgets: A New Approach",
        authors=[Author(name="Ada Lovelace", affiliation="Analytical Engines Inc.")],
        problem_statement="Widget classification is slow and inaccurate.",
        methodology="We propose a gradient-boosted widget classifier.",
        key_results="94.3% accuracy on WidgetBench.",
    )


def _orkg_contribution() -> ORKGContribution:
    return ORKGContribution(
        reasoning="r",
        paper_title="Gradient Widgets: A New Approach",
        research_problem="Widget classification accuracy.",
        contribution_label="Contribution 1",
        triples=[
            Triple(
                subject="Our Method",
                predicate="evaluates on",
                object="WidgetBench",
                evidence_quote="q",
            )
        ],
    )


def _idea_dossier() -> IdeaDossier:
    idea = StartupIdea(
        reasoning="r",
        name="WidgetSort AI",
        description="Automated widget sorting for manufacturers.",
        target_customer="Mid-size widget manufacturers",
        commercial_angle="Real-time gradient-boosted classification cuts sorting errors.",
        methodology_basis="We propose a gradient-boosted widget classifier.",
    )
    competitors = CompetitorLandscape(
        idea_name="WidgetSort AI",
        summary="A few players offer manual sorting tools; none use gradient boosting.",
        competitors=[Competitor(name="SortCo", description="Manual widget sorting SaaS.", url="https://sortco.example")],
        search_queries_used=["widget sorting software competitors"],
        sources=["https://sortco.example"],
    )
    tam = TAMEstimate(
        idea_name="WidgetSort AI",
        reasoning="Based on total widget manufacturers times average software spend.",
        assumptions=["10,000 widget manufacturers globally", "$5,000/year average spend"],
        calculation_code="print(10_000 * 5_000)",
        calculation_output="50000000\n",
        tam_usd=50_000_000.0,
        data_sources=["https://example.com/widget-industry-report"],
    )
    return IdeaDossier(idea=idea, competitors=competitors, tam=tam)


def test_report_includes_all_three_layers():
    essence = _essence()
    contribution = _orkg_contribution()
    dossier = _idea_dossier()

    report = render_investment_thesis_markdown(essence, contribution, [dossier])

    # Scientific summary layer
    assert "## Scientific Summary" in report
    assert "Ada Lovelace" in report
    assert "94.3% accuracy on WidgetBench." in report

    # ORKG layer
    assert "## ORKG Contribution" in report
    assert "| Our Method | evaluates on | WidgetBench |" in report

    # Venture Catalyst layer
    assert "### 1. WidgetSort AI" in report
    assert "SortCo" in report
    assert "**Estimated TAM: $50,000,000**" in report
    assert "print(10_000 * 5_000)" in report
    assert "50000000" in report
    assert "https://example.com/widget-industry-report" in report


def test_report_handles_multiple_ideas_with_incrementing_headers():
    essence = _essence()
    contribution = _orkg_contribution()
    dossier_a = _idea_dossier()
    dossier_b = _idea_dossier()
    dossier_b.idea.name = "SecondIdea"

    report = render_investment_thesis_markdown(essence, contribution, [dossier_a, dossier_b])

    assert "### 1. WidgetSort AI" in report
    assert "### 2. SecondIdea" in report
