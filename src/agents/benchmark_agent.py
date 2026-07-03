"""
Benchmark-extraction agent: the first stage of the Comparison Engine.

Reads a single paper's Markdown and pulls out its reported (dataset, metric,
value) results into a `PaperBenchmarks` object. The actual cross-paper
comparison logic (including how mismatched metric types like Accuracy vs.
F1-score are handled) lives in `src/comparison/comparison_engine.py` and is
deliberately kept out of the LLM entirely - see that module's docstring.

This agent uses the same two anti-hallucination mechanisms as
`src.agents.orkg_agent` (an ordered reasoning field, and a mandatory
verbatim `evidence_quote` per result that's programmatically verified
after the call) - see that module's docstring for the full explanation.
Numeric results are exactly the kind of content where hallucination is most
dangerous (a fabricated "94.3%" is worse than a fabricated adjective), so
grounding is enforced here just as strictly.
"""

import re
import sys

from src.agents.llm_client import DEFAULT_MODEL, get_instructor_client
from src.schemas.benchmarks import PaperBenchmarks

SYSTEM_PROMPT = """\
You are a scientific benchmarking analyst. Your job is to read a paper's \
Markdown and extract every reported (dataset, metric, value) result into a \
structured list.

Strict rules:
- Only extract numbers that are explicitly printed in the text or tables. \
Never estimate, round differently than the source, or fill in a result you \
don't see written down.
- Before listing any benchmarks, write out your reasoning: locate each \
results table or sentence that reports a number, and quote it, before \
turning it into a structured BenchmarkResult.
- Every BenchmarkResult must include an `evidence_quote` that is an exact, \
verbatim substring of the source Markdown (e.g. the table row or sentence \
it came from) - not a paraphrase.
- Normalize each result's `metric_type` to the closest match in the given \
enum (accuracy, f1, precision, recall, auc, bleu, rouge, rmse, mae, \
perplexity, other), but always preserve the metric's original wording in \
`metric_name_raw` and its original printed value (with unit/percent sign) \
in `value_raw`.
- Prefer omitting a result over guessing at a number.
"""


def _normalize(text: str) -> str:
    """Collapse whitespace and lowercase, for tolerant substring matching."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _filter_ungrounded_benchmarks(
    benchmarks: PaperBenchmarks, source_markdown: str
) -> PaperBenchmarks:
    """
    Drop any BenchmarkResult whose `evidence_quote` isn't in the source text.

    Data flow:
        Mirrors `src.agents.orkg_agent._filter_ungrounded_triples`: takes
        the `PaperBenchmarks` returned by the LLM call and the original
        paper Markdown, and keeps only the results whose `evidence_quote`
        is a verbatim (whitespace/case-normalized) substring of the source.
        This protects the Comparison Engine from ever tabulating a number
        the model invented.

    Args:
        benchmarks: The raw `PaperBenchmarks` returned by the LLM.
        source_markdown: The original Markdown the benchmarks were
            extracted from.

    Returns:
        The same `PaperBenchmarks`, with `benchmarks` filtered in place.
    """
    normalized_source = _normalize(source_markdown)
    grounded = [
        result
        for result in benchmarks.benchmarks
        if _normalize(result.evidence_quote) in normalized_source
    ]
    dropped = len(benchmarks.benchmarks) - len(grounded)
    if dropped:
        print(
            f"[benchmark_agent] dropped {dropped} result(s) with "
            "unverifiable evidence quotes",
            file=sys.stderr,
        )
    benchmarks.benchmarks = grounded
    return benchmarks


def extract_paper_benchmarks(
    markdown: str,
    paper_title: str | None = None,
    source_file: str | None = None,
    model: str = DEFAULT_MODEL,
    client=None,
) -> PaperBenchmarks:
    """
    Extract a paper's reported benchmark results as structured data.

    Data flow:
        1. Receives the paper's Markdown.
        2. Sends it to Claude via Instructor with `response_model=
           PaperBenchmarks`, enforcing the schema (including the
           `MetricType` enum) on the API response itself.
        3. Passes the result through `_filter_ungrounded_benchmarks` to
           drop any result whose quoted evidence isn't verifiably present
           in the source text.
        4. Returns the resulting `PaperBenchmarks`. `main.py`'s `compare`
           command collects one of these per input PDF and passes the
           list to `comparison_engine.build_comparison_table`.

    Args:
        markdown: The paper's content as Markdown.
        paper_title: Optional known title, passed as a hint to the model.
        source_file: Optional path/filename of the source PDF, stored on
            the result for traceability.
        model: Anthropic model id to use. Defaults to `DEFAULT_MODEL`.
        client: Optional pre-built Instructor client (mainly for testing);
            defaults to `get_instructor_client()`.

    Returns:
        A `PaperBenchmarks` with hallucination-filtered `benchmarks`.
    """
    client = client or get_instructor_client()

    user_prompt = f"""\
Paper title (if known): {paper_title or "unknown - extract it from the text"}

Read the paper below and extract every reported benchmark result (dataset, \
metric, value) into structured form.

--- PAPER MARKDOWN START ---
{markdown}
--- PAPER MARKDOWN END ---
"""

    benchmarks = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        response_model=PaperBenchmarks,
    )

    benchmarks.source_file = source_file
    return _filter_ungrounded_benchmarks(benchmarks, markdown)
