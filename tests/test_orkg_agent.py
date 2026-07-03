"""
Tests for the ORKG agent's grounding filter.

These tests use a fake Instructor client (no real Anthropic API call) so
they can run without `ANTHROPIC_API_KEY`. They exist to verify the one
piece of anti-hallucination logic that's actually testable without a live
LLM: `_filter_ungrounded_triples` must keep triples whose evidence quote is
genuinely in the source text, and drop ones whose quote is fabricated.
"""

from src.agents.orkg_agent import extract_orkg_contribution
from src.schemas.orkg import ORKGContribution, Triple

SOURCE_MARKDOWN = """\
# Our Method Improves Widget Classification

We propose a gradient-boosted approach to widget classification.

We evaluate our method on the WidgetBench dataset and achieve 94.3% accuracy.
"""


class _FakeMessages:
    def __init__(self, response: ORKGContribution):
        self._response = response

    def create(self, **kwargs):
        return self._response


class _FakeInstructorClient:
    """Stands in for an instructor.Instructor client, returning a canned response."""

    def __init__(self, response: ORKGContribution):
        self.messages = _FakeMessages(response)


def test_grounded_triple_is_kept_and_ungrounded_triple_is_dropped():
    canned_response = ORKGContribution(
        reasoning="Found two facts: the approach, and the benchmark result.",
        paper_title="Our Method Improves Widget Classification",
        research_problem="Widget classification accuracy.",
        contribution_label="Contribution 1",
        triples=[
            Triple(
                subject="Our Method",
                predicate="has approach",
                object="gradient-boosted approach",
                evidence_quote="We propose a gradient-boosted approach to widget classification.",
            ),
            Triple(
                subject="Our Method",
                predicate="achieves",
                # This triple's evidence quote is fabricated - it does not
                # appear anywhere in SOURCE_MARKDOWN.
                object="99.9% accuracy on FakeBench",
                evidence_quote="We achieve 99.9% accuracy on FakeBench, a dataset we invented.",
            ),
        ],
    )
    fake_client = _FakeInstructorClient(canned_response)

    result = extract_orkg_contribution(
        SOURCE_MARKDOWN,
        paper_title="Our Method Improves Widget Classification",
        source_file="widget.pdf",
        client=fake_client,
    )

    assert len(result.triples) == 1
    assert result.triples[0].object == "gradient-boosted approach"
    assert result.source_file == "widget.pdf"


def test_grounding_check_is_whitespace_and_case_tolerant():
    """
    Docling-produced Markdown often reflows whitespace/line-breaks compared
    to how a model might quote it back; the grounding check should still
    match as long as the words themselves are a genuine substring.
    """
    canned_response = ORKGContribution(
        reasoning="r",
        paper_title="Our Method Improves Widget Classification",
        research_problem="p",
        contribution_label="Contribution 1",
        triples=[
            Triple(
                subject="Our Method",
                predicate="evaluates on",
                object="WidgetBench",
                evidence_quote="we evaluate our method on the widgetbench dataset",
            ),
        ],
    )
    fake_client = _FakeInstructorClient(canned_response)

    result = extract_orkg_contribution(SOURCE_MARKDOWN, client=fake_client)

    assert len(result.triples) == 1
