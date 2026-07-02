"""
Markdown -> ScientificEssence extraction.

This module is intentionally a *stub*. Reliably pulling the problem
statement, methodology, and key results out of a paper's Markdown is a
reasoning task, not a parsing task - it belongs to an LLM call (e.g. "read
this Markdown and return JSON matching the ScientificEssence schema"), which
is out of scope for this initial project-scaffolding task.

What this module does today is a cheap, dependency-free heuristic extractor
that lets `main.py` produce a real `ScientificEssence` JSON file end-to-end
(so the whole pipeline is runnable and testable), populated with whatever it
can confidently pull from the document structure alone: the title. Every
other field is left as a clearly-labeled placeholder.

When VentureGraph adds LLM-based extraction, this function's signature can
stay the same - only the implementation needs to change to call out to an
LLM with the schema as its response format.
"""

from src.schemas.scientific_essence import ScientificEssence


def extract_essence_stub(markdown: str, source_file: str | None = None) -> ScientificEssence:
    """
    Produce a best-effort ScientificEssence from a paper's Markdown.

    Data flow:
        1. Receives the full Markdown text produced by
           `src.parser.pdf_parser.convert_pdf_to_markdown`.
        2. Looks at the first Markdown heading line (a line starting with
           `# `) and uses it as `paper_title`, since Docling's layout model
           reliably tags the paper's title as the top-level heading.
        3. Every field that requires actual comprehension of the paper's
           content (problem statement, methodology, key results, authors)
           is filled with an explicit placeholder string rather than a
           guess, so downstream consumers can tell "not yet extracted"
           apart from "extracted as empty".
        4. Returns a fully-populated `ScientificEssence` instance, ready to
           be serialized to JSON by `main.py`.

    Args:
        markdown: The paper's content as Markdown (from the PDF parser).
        source_file: Optional path/filename of the original PDF, stored on
            the resulting essence for traceability.

    Returns:
        A `ScientificEssence` instance. Only `paper_title` is derived from
        the document; the remaining required fields are populated with a
        "[NOT YET EXTRACTED]" placeholder pending LLM-based extraction.
    """
    placeholder = "[NOT YET EXTRACTED - requires LLM-based reading comprehension]"

    title = placeholder
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped.removeprefix("# ").strip()
            break

    return ScientificEssence(
        paper_title=title,
        authors=[],
        problem_statement=placeholder,
        methodology=placeholder,
        key_results=placeholder,
        source_file=source_file,
    )
