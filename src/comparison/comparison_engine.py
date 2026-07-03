"""
Comparison Engine: builds a Markdown table comparing two or more papers'
benchmark results.

This module is deliberately plain, deterministic Python with no LLM calls.
By the time a `PaperBenchmarks` object reaches this module, an LLM (see
`src.agents.benchmark_agent`) has already done the hard, judgment-heavy work
of reading each paper and structuring its results; grouping and formatting
those already-structured results into a table is a mechanical operation
that doesn't need - and shouldn't need - another model call. Keeping it
LLM-free also makes it trivially unit-testable and reproducible: the same
input always produces the same table.

Handling papers that use different units of measurement
----------------------------------------------------------
This is the interesting design problem: if Paper A reports 94.3% Accuracy
on a dataset and Paper B reports 0.87 F1-score on the *same* dataset, what
should the table show?

The engine deliberately does **not** try to convert one metric into the
other. Accuracy and F1-score measure different things (F1 is the harmonic
mean of precision and recall on a chosen positive class; accuracy is
overall correctness across all classes) and are only interconvertible if
you also know the class balance and the full confusion matrix - which a
benchmark table almost never reports. Silently converting or averaging them
would be fabricating a number that isn't in either paper, which is exactly
the kind of hallucination this whole system is designed to avoid.

Instead, `build_comparison_table` does the following for every dataset row:
    1. Groups all `BenchmarkResult`s across all input papers by dataset
       name (case/whitespace-normalized).
    2. For each paper's column, shows every metric that paper reported for
       that dataset, labeled by metric type (e.g. "f1: 0.87"), so nothing
       is silently dropped or merged.
    3. Computes the *set* of metric types reported for that dataset across
       all papers. If that set has more than one member, the row is
       flagged with a "⚠" marker and a footnote is appended below the
       table explaining, by name, which metrics don't line up and why they
       can't be compared numerically.

This makes the mismatch visible and explicit rather than hidden behind a
misleading single number - the human reader decides what to make of it,
with both original values in front of them.
"""

from collections import defaultdict

from src.schemas.benchmarks import BenchmarkResult, PaperBenchmarks


def _format_result(result: BenchmarkResult) -> str:
    """Render one BenchmarkResult as a table-cell fragment, e.g. 'f1: 0.87'."""
    return f"{result.metric_type.value}: {result.value_raw}"


def build_comparison_table(papers: list[PaperBenchmarks]) -> str:
    """
    Build a Markdown table comparing benchmark results across papers.

    Data flow:
        1. Takes a list of `PaperBenchmarks` (one per paper, produced by
           `src.agents.benchmark_agent.extract_paper_benchmarks`).
        2. Groups every `BenchmarkResult` across all papers by dataset name
           (normalized for whitespace/case, but displayed using the first
           spelling seen).
        3. For each dataset, builds one table row with one column per
           paper, listing every metric that paper reported for that
           dataset (there can be more than one, e.g. both Accuracy and F1).
        4. If, for a given dataset, the papers collectively used more than
           one distinct `MetricType`, marks that row with "⚠" and records a
           footnote explaining the mismatch (see module docstring for the
           reasoning behind not converting between metrics).
        5. Returns the assembled Markdown table (plus any footnotes) as a
           single string, ready to write to a `.md` file or print.

    Args:
        papers: A list of at least two `PaperBenchmarks` to compare.

    Returns:
        A Markdown string: a table with one row per dataset and one column
        per paper, followed by a "Metric mismatches" footnote section if
        any dataset had inconsistent metric types across papers.

    Raises:
        ValueError: If fewer than two papers are provided.
    """
    if len(papers) < 2:
        raise ValueError("Comparison requires at least two papers.")

    # normalized dataset key -> paper_title -> list[BenchmarkResult]
    datasets: dict[str, dict[str, list[BenchmarkResult]]] = defaultdict(lambda: defaultdict(list))
    dataset_display_names: dict[str, str] = {}

    for paper in papers:
        for result in paper.benchmarks:
            key = result.dataset.strip().lower()
            dataset_display_names.setdefault(key, result.dataset.strip())
            datasets[key][paper.paper_title].append(result)

    paper_titles = [paper.paper_title for paper in papers]

    header = "| Dataset | " + " | ".join(paper_titles) + " |"
    separator = "|---|" + "|".join(["---"] * len(paper_titles)) + "|"
    rows = [header, separator]
    footnotes: list[str] = []

    for key in sorted(datasets.keys()):
        display_name = dataset_display_names[key]
        results_by_paper = datasets[key]

        metric_types_seen = {
            result.metric_type
            for results in results_by_paper.values()
            for result in results
        }
        mismatched = len(metric_types_seen) > 1

        row_label = f"{display_name} ⚠" if mismatched else display_name
        cells = []
        for title in paper_titles:
            results = results_by_paper.get(title, [])
            cells.append("; ".join(_format_result(r) for r in results) if results else "—")
        rows.append(f"| {row_label} | " + " | ".join(cells) + " |")

        if mismatched:
            metric_names = ", ".join(sorted(metric_type.value for metric_type in metric_types_seen))
            footnotes.append(
                f"- **{display_name}**: papers report different metric types "
                f"({metric_names}). These are not numerically comparable "
                f"without additional information (e.g. Accuracy and F1-score "
                f"measure different things and can't be converted between "
                f"each other without the underlying confusion matrix) - "
                f"values are shown side-by-side above, not merged or converted."
            )

    table = "\n".join(rows)
    if footnotes:
        table += "\n\n**⚠ Metric mismatches:**\n" + "\n".join(footnotes)
    return table


def generate_comparison_report(papers: list[PaperBenchmarks]) -> str:
    """
    Build a full Markdown comparison report: a title, the list of papers
    being compared, and the comparison table.

    Data flow:
        Thin wrapper around `build_comparison_table` that adds a heading
        and a bullet list naming each paper (with its source file, if
        known), so the output is a self-contained, readable Markdown
        document rather than a bare table. This is what `main.py`'s
        `compare` command writes to disk.

    Args:
        papers: A list of at least two `PaperBenchmarks` to compare.

    Returns:
        A complete Markdown document as a string.
    """
    lines = ["# Paper Comparison", ""]
    for paper in papers:
        suffix = f" ({paper.source_file})" if paper.source_file else ""
        lines.append(f"- **{paper.paper_title}**{suffix}")
    lines.append("")
    lines.append(build_comparison_table(papers))
    return "\n".join(lines)
