"""
ORKG specialist agent: extracts a paper's contribution as Subject-Predicate-
Object triples, structured to match the Open Research Knowledge Graph
(ORKG) contribution model.

Chain-of-Thought strategy (how this avoids hallucinated triples)
------------------------------------------------------------------
Two mechanisms work together, both enforced through the Pydantic response
schema rather than prompt wording alone (since Instructor validates the
model's output against `ORKGContribution` and can reprompt on failure):

1. Ordered "reason-then-answer" fields. `ORKGContribution.reasoning` is
   declared *before* `triples` in the schema (see `src/schemas/orkg.py`).
   Claude's tool-calling fills a JSON object's fields in the order they're
   declared in the schema, and generation is autoregressive - so the model
   is forced to write out, for each candidate contribution, which
   sentence(s) in the paper support it, *before* it is allowed to commit to
   the structured `triples` list. This is the standard "let it think
   before it answers" Chain-of-Thought trick, implemented via schema field
   order instead of a free-text "think step by step" instruction the model
   could otherwise skip past.

2. Mandatory verbatim grounding + programmatic verification. Every `Triple`
   must carry an `evidence_quote` - a direct quote from the source
   Markdown. We do not just trust the model on this: after the LLM call
   returns, `_filter_ungrounded_triples()` checks each quote against the
   actual source text (whitespace-normalized substring match) and silently
   drops any triple whose evidence cannot be found verbatim in the paper.
   This turns "please don't hallucinate" from a prompt suggestion into a
   deterministic, code-enforced filter: a triple built on a fabricated
   quote cannot survive into the final result, no matter how plausible it
   reads.

The system prompt below reinforces both mechanisms explicitly, and sets
temperature=0 in the API call to keep extraction as deterministic/factual
as possible.
"""

import re
import sys

from src.agents.llm_client import DEFAULT_MODEL, get_instructor_client
from src.schemas.orkg import ORKGContribution

SYSTEM_PROMPT = """\
You are an ORKG (Open Research Knowledge Graph) curation specialist. Your \
job is to read a scientific paper's Markdown and extract its contribution(s) \
as Subject-Predicate-Object triples suitable for a research knowledge graph.

Strict rules:
- Only extract facts that are explicitly stated in the provided text. Never \
infer, assume, or add outside knowledge about the topic, even if you \
recognize the paper or the field.
- Before listing any triples, write out your reasoning: for each fact you \
plan to turn into a triple, quote the exact sentence(s) from the text that \
support it.
- Every triple must include an `evidence_quote` that is an exact, verbatim \
substring of the source Markdown - not a paraphrase. If you cannot find a \
verbatim quote to support a fact, do not include it as a triple.
- Prefer omitting a triple over guessing. It is far better to return fewer, \
well-grounded triples than to pad the list with speculative ones.
- Keep subjects/predicates/objects concise (a few words), not full sentences.
"""


def _normalize(text: str) -> str:
    """Collapse whitespace and lowercase, for tolerant substring matching."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _filter_ungrounded_triples(
    contribution: ORKGContribution, source_markdown: str
) -> ORKGContribution:
    """
    Drop any triple whose `evidence_quote` isn't actually in the source text.

    Data flow:
        Takes the `ORKGContribution` returned by the LLM call and the
        original paper Markdown it was extracted from. Normalizes both
        (collapsing whitespace, lowercasing) and keeps only the triples
        whose `evidence_quote` is a substring of the normalized source.
        This is the programmatic half of the anti-hallucination strategy
        described in this module's docstring - it runs regardless of how
        confident or fluent the model's output looks.

    Args:
        contribution: The raw `ORKGContribution` returned by the LLM.
        source_markdown: The original Markdown the contribution was
            extracted from, used as the ground truth to verify quotes
            against.

    Returns:
        The same `ORKGContribution`, with `triples` filtered in place to
        only those whose evidence quote is verifiably present in the text.
    """
    normalized_source = _normalize(source_markdown)
    grounded = [
        triple
        for triple in contribution.triples
        if _normalize(triple.evidence_quote) in normalized_source
    ]
    dropped = len(contribution.triples) - len(grounded)
    if dropped:
        print(
            f"[orkg_agent] dropped {dropped} triple(s) with unverifiable "
            "evidence quotes",
            file=sys.stderr,
        )
    contribution.triples = grounded
    return contribution


def extract_orkg_contribution(
    markdown: str,
    paper_title: str | None = None,
    source_file: str | None = None,
    model: str = DEFAULT_MODEL,
    client=None,
) -> ORKGContribution:
    """
    Extract an ORKG-compatible contribution (as SPO triples) from a paper.

    Data flow:
        1. Receives the paper's Markdown (from
           `src.parser.pdf_parser.convert_pdf_to_markdown`).
        2. Sends it to Claude via Instructor with `response_model=
           ORKGContribution`, so the API call itself enforces the schema
           (field types, required fields) and reprompts on a validation
           failure.
        3. Passes the returned `ORKGContribution` through
           `_filter_ungrounded_triples` to drop any triple whose quoted
           evidence isn't actually present in the source text.
        4. Returns the resulting `ORKGContribution`, ready to be
           serialized to JSON (e.g. by `main.py`) or pushed into ORKG.

    Args:
        markdown: The paper's content as Markdown.
        paper_title: Optional known title, passed to the model as a hint
            (it will still confirm/extract the title itself).
        source_file: Optional path/filename of the source PDF, stored on
            the result for traceability.
        model: Anthropic model id to use. Defaults to `DEFAULT_MODEL`.
        client: Optional pre-built Instructor client (mainly for testing);
            defaults to `get_instructor_client()`.

    Returns:
        An `ORKGContribution` with hallucination-filtered `triples`.
    """
    client = client or get_instructor_client()

    user_prompt = f"""\
Paper title (if known): {paper_title or "unknown - extract it from the text"}

Read the paper below and extract its scientific contribution as ORKG-style \
Subject-Predicate-Object triples (e.g. subject="Our Method", \
predicate="evaluates on", object="ImageNet"). Cover things like: the \
approach/method used, the datasets/benchmarks used, the metrics reported, \
and the key results, each as its own triple.

--- PAPER MARKDOWN START ---
{markdown}
--- PAPER MARKDOWN END ---
"""

    contribution = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        response_model=ORKGContribution,
    )

    contribution.source_file = source_file
    return _filter_ungrounded_triples(contribution, markdown)
