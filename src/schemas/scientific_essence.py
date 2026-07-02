"""
Pydantic schema for the "Scientific Essence" of a paper.

This schema is the core data contract of VentureGraph: it is the distilled,
structured representation that every paper gets reduced to before it feeds
into downstream reasoning (ORKG contribution generation, startup-opportunity
scoring, comparison tables across papers, etc.). Keeping this as a strict
Pydantic model - rather than a loose dict - means every producer (an LLM
extractor, a human editor, a future API endpoint) and every consumer agree
on exactly what fields exist, what type they are, and what "required" means.
"""

from pydantic import BaseModel, Field


class Author(BaseModel):
    """
    A single author of a scientific paper.

    Data flow:
        Instances of this model are collected into the `authors` list on
        `ScientificEssence`. They typically come from an LLM (or parser)
        reading the author block at the top of the paper's Markdown, right
        after the title.
    """

    name: str = Field(
        ...,
        description="Full name of the author as printed on the paper, e.g. 'Jane A. Doe'.",
    )
    affiliation: str | None = Field(
        default=None,
        description=(
            "Author's institutional affiliation (university, lab, or company), "
            "if it is stated in the paper. None if not available."
        ),
    )


class ScientificEssence(BaseModel):
    """
    The distilled scientific essence of a single paper.

    Data flow:
        1. `src/parser/pdf_parser.py` converts a source PDF into Markdown.
        2. An extraction step (see `src/extractor/essence_extractor.py`)
           reads that Markdown and produces one `ScientificEssence`
           instance - either via an LLM call that is instructed to return
           JSON matching this schema, or via simpler heuristics.
        3. `main.py` serializes the resulting instance with
           `.model_dump_json()` and writes/prints it, so it can be stored,
           diffed against other papers, or handed to the next VentureGraph
           stage (e.g. ORKG contribution mapping or startup-idea scoring).

    Every field below is intentionally a plain, LLM-friendly type (strings
    and a list of a simple sub-model) so that this schema can double as a
    JSON-schema prompt/response contract for structured LLM extraction.
    """

    paper_title: str = Field(
        ...,
        description="The full title of the paper, exactly as it appears on the paper.",
    )
    authors: list[Author] = Field(
        default_factory=list,
        description="Ordered list of the paper's authors, as they appear on the paper.",
    )
    problem_statement: str = Field(
        ...,
        description=(
            "A concise summary of the problem or research gap the paper "
            "addresses - what motivated the work, and why it matters."
        ),
    )
    methodology: str = Field(
        ...,
        description=(
            "A concise summary of the approach, method, model, or experimental "
            "design the authors used to address the problem statement."
        ),
    )
    key_results: str = Field(
        ...,
        description=(
            "A concise summary of the paper's main findings/results - the "
            "headline numbers, outcomes, or conclusions the authors report."
        ),
    )

    source_file: str | None = Field(
        default=None,
        description=(
            "Filename or path of the source PDF this essence was extracted "
            "from, kept for traceability back to the original document."
        ),
    )
