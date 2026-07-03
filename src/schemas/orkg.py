"""
Pydantic schema for ORKG-style contribution triples.

The Open Research Knowledge Graph (ORKG) models a paper's contribution as a
set of statements shaped like a knowledge-graph triple: a Subject (usually
the contribution itself, or a resource it introduces), a Predicate (an ORKG
property such as "has approach", "evaluates on", "employs metric"), and an
Object (a value or another resource, e.g. a dataset name or a result). This
schema captures that shape in a form an LLM can reliably fill in and that
code can validate.
"""

from pydantic import BaseModel, Field


class Triple(BaseModel):
    """
    A single Subject-Predicate-Object statement describing part of a paper's
    contribution, in a form compatible with ORKG's statement model.

    Data flow:
        Produced by `src.agents.orkg_agent.extract_orkg_contribution`, one
        per atomic fact the LLM finds in the paper (e.g. "Our Method" -
        "evaluates on" - "ImageNet"). Collected into the `triples` list on
        `ORKGContribution`.
    """

    subject: str = Field(
        ...,
        description=(
            "The subject of the statement - typically the paper's "
            "contribution, method, or model, e.g. 'Our Method' or 'The "
            "proposed model'."
        ),
    )
    predicate: str = Field(
        ...,
        description=(
            "The relationship/property connecting subject to object, "
            "phrased as an ORKG-style predicate, e.g. 'has approach', "
            "'evaluates on dataset', 'employs metric', 'achieves', "
            "'compared against'."
        ),
    )
    object: str = Field(
        ...,
        description=(
            "The object of the statement - the value or resource the "
            "predicate points to, e.g. 'gradient boosting', 'ImageNet', "
            "'94.3% accuracy'."
        ),
    )
    evidence_quote: str = Field(
        ...,
        description=(
            "A verbatim quote copied exactly from the source Markdown that "
            "directly supports this triple. Must be an exact substring of "
            "the source text - do not paraphrase or summarize here. This "
            "field exists purely so the triple's grounding can be checked "
            "programmatically after extraction."
        ),
    )


class ORKGContribution(BaseModel):
    """
    A paper's scientific contribution(s), expressed as ORKG-compatible
    triples, plus the reasoning trace that produced them.

    Data flow:
        1. `src.agents.orkg_agent.extract_orkg_contribution` sends a
           paper's Markdown to Claude via Instructor, which validates the
           model's JSON response against this schema (retrying on schema
           violations).
        2. The `reasoning` field is declared *before* `triples` so the
           model is forced to write out its evidence-gathering reasoning
           before committing to structured triples (see the
           Chain-of-Thought explanation in `orkg_agent.py`).
        3. After the LLM call, `orkg_agent.py` filters `triples` to drop
           any whose `evidence_quote` cannot be found verbatim in the
           source Markdown, then returns the resulting `ORKGContribution`.
        4. `main.py` serializes the result with `.model_dump_json()` for
           storage, or hands it to ORKG-export tooling downstream.
    """

    reasoning: str = Field(
        ...,
        description=(
            "Step-by-step reasoning written BEFORE the triples: for each "
            "contribution you plan to extract, quote the exact sentence(s) "
            "in the source Markdown that justify it. Do this for every "
            "triple you intend to produce. If you cannot find explicit "
            "textual support for something, say so here and leave it out "
            "of the triples list rather than guessing."
        ),
    )
    paper_title: str = Field(..., description="The title of the paper, as it appears in the text.")
    research_problem: str = Field(
        ...,
        description="The research problem this contribution addresses, in the paper's own terms.",
    )
    contribution_label: str = Field(
        ...,
        description="A short label for this contribution, e.g. 'Contribution 1' or a descriptive name.",
    )
    triples: list[Triple] = Field(
        default_factory=list,
        description="The Subject-Predicate-Object triples that make up this contribution.",
    )
    source_file: str | None = Field(
        default=None,
        description="Filename/path of the source PDF this contribution was extracted from.",
    )
