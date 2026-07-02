#!/usr/bin/env python3
"""
VentureGraph CLI.

Usage:
    python main.py parse path/to/paper.pdf [--output-dir output]
    python main.py orkg path/to/paper.pdf [--output-dir output]
    python main.py compare path/to/paper_a.pdf path/to/paper_b.pdf [...] [--output-dir output]

Subcommands:
    parse    - PDF -> Markdown + ScientificEssence JSON (Docling only, no LLM).
    orkg     - PDF -> Markdown, then extract ORKG-style Subject-Predicate-
               Object contribution triples via the ORKG specialist agent
               (requires ANTHROPIC_API_KEY).
    compare  - Two or more PDFs -> Markdown, then extract each paper's
               benchmark results and build a Markdown comparison table via
               the Comparison Engine (requires ANTHROPIC_API_KEY for the
               per-paper extraction step; the comparison itself is a plain
               deterministic function with no LLM call).
"""

import argparse
import sys
from pathlib import Path

from src.agents.benchmark_agent import extract_paper_benchmarks
from src.agents.orkg_agent import extract_orkg_contribution
from src.comparison.comparison_engine import generate_comparison_report
from src.extractor.essence_extractor import extract_essence_stub
from src.parser.pdf_parser import convert_pdf_to_markdown
from src.schemas.benchmarks import PaperBenchmarks
from src.schemas.orkg import ORKGContribution
from src.schemas.scientific_essence import ScientificEssence


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the VentureGraph CLI.

    Data flow:
        Takes the raw CLI argument list (or `sys.argv` if `argv` is None)
        and returns a populated `argparse.Namespace`. The `command`
        attribute (set by the chosen subparser) tells `main()` which of
        `run_parse`, `run_orkg`, or `run_compare` to call; the rest of the
        namespace holds that subcommand's own arguments (`pdf_path`(s) and
        `output_dir`).

    Args:
        argv: Optional explicit argument list, mainly for testing. Defaults
            to reading from `sys.argv` when None.

    Returns:
        Parsed arguments as an `argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description=(
            "VentureGraph: turn scientific PDFs into clean Markdown, "
            "structured essences, ORKG contribution triples, and "
            "cross-paper benchmark comparisons."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_p = subparsers.add_parser(
        "parse", help="Convert a PDF to Markdown and a ScientificEssence JSON (no LLM)."
    )
    parse_p.add_argument("pdf_path", type=Path, help="Path to the input scientific PDF.")
    parse_p.add_argument(
        "--output-dir", type=Path, default=Path("output"), help="Output directory (default: ./output)."
    )

    orkg_p = subparsers.add_parser(
        "orkg", help="Extract ORKG-style contribution triples from a PDF (uses Claude)."
    )
    orkg_p.add_argument("pdf_path", type=Path, help="Path to the input scientific PDF.")
    orkg_p.add_argument(
        "--output-dir", type=Path, default=Path("output"), help="Output directory (default: ./output)."
    )

    compare_p = subparsers.add_parser(
        "compare",
        help="Compare benchmark results across two or more PDFs (uses Claude for per-paper extraction).",
    )
    compare_p.add_argument(
        "pdf_paths", type=Path, nargs="+", help="Paths to two or more scientific PDFs to compare."
    )
    compare_p.add_argument(
        "--output-dir", type=Path, default=Path("output"), help="Output directory (default: ./output)."
    )

    return parser.parse_args(argv)


def run_parse(pdf_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """
    Run the PDF -> Markdown -> ScientificEssence pipeline for one file.

    Data flow:
        1. Converts `pdf_path` to Markdown via Docling
           (`convert_pdf_to_markdown`).
        2. Writes that Markdown to `<output_dir>/<pdf stem>.md`.
        3. Extracts a `ScientificEssence` from the Markdown
           (`extract_essence_stub` - a heuristic stub, no LLM call).
        4. Writes the essence as pretty-printed JSON to
           `<output_dir>/<pdf stem>.essence.json`.

    Args:
        pdf_path: Path to the input PDF file.
        output_dir: Directory where the two output files will be written
            (created if it doesn't already exist).

    Returns:
        A tuple `(markdown_path, json_path)` pointing to the two files that
        were written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown = convert_pdf_to_markdown(pdf_path)
    markdown_path = output_dir / f"{pdf_path.stem}.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    essence: ScientificEssence = extract_essence_stub(markdown, source_file=str(pdf_path))
    json_path = output_dir / f"{pdf_path.stem}.essence.json"
    json_path.write_text(essence.model_dump_json(indent=2), encoding="utf-8")

    return markdown_path, json_path


def run_orkg(pdf_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """
    Run the PDF -> Markdown -> ORKGContribution pipeline for one file.

    Data flow:
        1. Converts `pdf_path` to Markdown via Docling.
        2. Writes that Markdown to `<output_dir>/<pdf stem>.md`, so the
           triples can always be cross-checked against the exact text they
           were extracted from.
        3. Calls `src.agents.orkg_agent.extract_orkg_contribution`, which
           sends the Markdown to Claude (via Instructor) and returns a
           hallucination-filtered `ORKGContribution`.
        4. Writes that contribution as pretty-printed JSON to
           `<output_dir>/<pdf stem>.orkg.json`.

    Args:
        pdf_path: Path to the input PDF file.
        output_dir: Directory where the two output files will be written.

    Returns:
        A tuple `(markdown_path, json_path)` pointing to the two files that
        were written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown = convert_pdf_to_markdown(pdf_path)
    markdown_path = output_dir / f"{pdf_path.stem}.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    contribution: ORKGContribution = extract_orkg_contribution(markdown, source_file=str(pdf_path))
    json_path = output_dir / f"{pdf_path.stem}.orkg.json"
    json_path.write_text(contribution.model_dump_json(indent=2), encoding="utf-8")

    return markdown_path, json_path


def run_compare(pdf_paths: list[Path], output_dir: Path) -> Path:
    """
    Run the multi-PDF -> Markdown -> PaperBenchmarks -> comparison pipeline.

    Data flow:
        1. For each PDF in `pdf_paths`: converts it to Markdown via
           Docling, then calls
           `src.agents.benchmark_agent.extract_paper_benchmarks` to get a
           hallucination-filtered `PaperBenchmarks` for that paper.
        2. Passes the full list of `PaperBenchmarks` to
           `src.comparison.comparison_engine.generate_comparison_report`,
           which deterministically groups results by dataset and builds
           the Markdown comparison table (flagging any dataset where
           papers used different metric types, e.g. Accuracy vs. F1).
        3. Writes the resulting Markdown report to
           `<output_dir>/comparison.md`.

    Args:
        pdf_paths: Paths to two or more input PDFs to compare.
        output_dir: Directory where the comparison report will be written.

    Returns:
        The path to the written `comparison.md` file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_benchmarks: list[PaperBenchmarks] = []
    for pdf_path in pdf_paths:
        markdown = convert_pdf_to_markdown(pdf_path)
        (output_dir / f"{pdf_path.stem}.md").write_text(markdown, encoding="utf-8")
        benchmarks = extract_paper_benchmarks(markdown, source_file=str(pdf_path))
        all_benchmarks.append(benchmarks)

    report = generate_comparison_report(all_benchmarks)
    report_path = output_dir / "comparison.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    """
    CLI entry point: parse arguments, dispatch to the right subcommand.

    Data flow:
        Calls `parse_args()` to get the chosen subcommand and its
        arguments, validates that the given PDF path(s) exist, dispatches
        to `run_parse`, `run_orkg`, or `run_compare`, and prints the
        resulting output file path(s). Exits with a non-zero status and an
        error message if any input PDF is missing.
    """
    args = parse_args()

    pdf_paths = args.pdf_paths if args.command == "compare" else [args.pdf_path]
    for pdf_path in pdf_paths:
        if not pdf_path.is_file():
            print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
            sys.exit(1)

    if args.command == "parse":
        markdown_path, json_path = run_parse(args.pdf_path, args.output_dir)
        print(f"Markdown written to: {markdown_path}")
        print(f"Essence JSON written to: {json_path}")
    elif args.command == "orkg":
        markdown_path, json_path = run_orkg(args.pdf_path, args.output_dir)
        print(f"Markdown written to: {markdown_path}")
        print(f"ORKG contribution JSON written to: {json_path}")
    elif args.command == "compare":
        if len(args.pdf_paths) < 2:
            print("Error: 'compare' requires at least two PDFs.", file=sys.stderr)
            sys.exit(1)
        report_path = run_compare(args.pdf_paths, args.output_dir)
        print(f"Comparison report written to: {report_path}")


if __name__ == "__main__":
    main()
